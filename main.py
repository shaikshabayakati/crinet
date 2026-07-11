"""
CriNet — Real-Time Scam Detection Backend
==========================================
Routes:
  Phase 1  → POST /call/start     — Vobiz outbound call trigger
  Phase 1  → POST /answer         — Vobiz answer-url webhook  (returns XML)
  Phase 2  → WS  /media-stream    — Vobiz audio forking websocket
  Phase 3  → (internal)           — Gemini Live audio bridge + scam detection
  Phase 4  → WS  /alerts          — Push alerts to the phone "app" (HTML page)
  Util     → GET /                — Health check
  Util     → GET /app             — Serve alert.html to the phone browser
"""

import asyncio
import base64
import json
import logging
import os
import struct
from contextlib import asynccontextmanager
from typing import Optional

from pathlib import Path
import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

# ─── Config ───────────────────────────────────────────────────────────────────
load_dotenv()

VOBIZ_AUTH_ID    = os.getenv("VOBIZ_AUTH_ID", "")
VOBIZ_AUTH_TOKEN = os.getenv("VOBIZ_AUTH_TOKEN", "")
VOBIZ_PHONE      = os.getenv("VOBIZ_PHONE_NUMBER", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
PUBLIC_HOST      = os.getenv("PUBLIC_HOST", "").strip().rstrip("/")

VOBIZ_API_BASE   = "https://api.vobiz.ai/api/v1"
GEMINI_WS_URL    = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)
GEMINI_MODEL     = "models/gemini-3.1-flash-live-preview"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("crinet")

# ─── Module-level state (survives --reload) ──────────────────────────────────
# Maps scammer-number-variants → victim-number
pending_victims: dict[str, str] = {}
# Maps scammer CallUUID → victim number (set in /answer, read in /dial-result)
active_call_uuids: dict[str, str] = {}

# ─── Alert broadcaster ────────────────────────────────────────────────────────
alert_subscribers: list[WebSocket] = []

async def broadcast_alert(payload: dict):
    dead = []
    for ws in alert_subscribers:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        alert_subscribers.remove(ws)

# ─── Gemini system prompt & tool declaration ──────────────────────────────────
SYSTEM_PROMPT = """
You are silently monitoring a live phone call. NEVER speak or generate audio output.
Watch for these scam patterns:
  • OTP / PIN / password requests
  • Gift-card or cryptocurrency payment demands
  • Impersonation of bank, police, government, or tech-support
  • Artificial urgency, threats, or countdown pressure
  • Requests to install AnyDesk, TeamViewer, or other remote-access software
  • Pressure to switch to WhatsApp, Telegram, or another platform

The moment you detect ANY of the above, call flag_scam_alert IMMEDIATELY.
Do not wait for the call to end. Do not ask clarifying questions. Just call the function.
"""

GEMINI_SETUP_MSG = {
    "setup": {
        "model": GEMINI_MODEL,
        "responseModalities": ["TEXT"],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "tools": [{
            "function_declarations": [{
                "name": "flag_scam_alert",
                "description": "Call the instant a scam pattern appears.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "otp_request",
                                "gift_card",
                                "impersonation",
                                "urgency",
                                "remote_access",
                                "other",
                            ],
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0–1.0 confidence score",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of the detected pattern",
                        },
                    },
                    "required": ["category", "confidence"],
                },
            }]
        }],
    }
}

# ─── Gemini Live session ──────────────────────────────────────────────────────
class GeminiSession:
    """Manages one Gemini Live websocket session per phone call."""

    def __init__(self):
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False
        self._recv_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None

    async def start(self):
        log.info("Connecting to Gemini Live …")
        try:
            self._ws = await websockets.connect(
                GEMINI_WS_URL,
                ping_interval=20,
                ping_timeout=10,
            )
            # 1 — send setup
            setup_json = json.dumps(GEMINI_SETUP_MSG)
            log.info("Sending Gemini setup: %s", setup_json[:300])
            await self._ws.send(setup_json)

            # 2 — wait for setupComplete
            raw = await self._ws.recv()
            resp = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
            log.info("Gemini setup response: %s", json.dumps(resp)[:500])

            if "setupComplete" not in resp:
                log.error("Gemini did NOT return setupComplete! Full response: %s", resp)
                raise RuntimeError(f"Gemini setup failed: {resp}")

            self._running = True
            self._recv_task = asyncio.create_task(self._recv_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            log.info("Gemini Live session ready — streaming audio now.")
        except Exception as exc:
            log.error("Gemini connection failed: %s", exc, exc_info=True)
            raise

    async def push_audio(self, pcm_le_bytes: bytes):
        """Queue little-endian PCM16 audio to be streamed to Gemini."""
        await self._audio_queue.put(pcm_le_bytes)

    async def stop(self):
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._send_task:
            self._send_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _send_loop(self):
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                audio_b64 = base64.b64encode(chunk).decode()
                msg = {
                    "realtime_input": {
                        "media_chunks": [{
                            "mime_type": "audio/pcm;rate=16000",
                            "data": audio_b64,
                        }]
                    }
                }
                await self._ws.send(json.dumps(msg))
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                log.error("Gemini send error: %s", exc)
                break

    async def _recv_loop(self):
        while self._running:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                await self._handle_server_msg(raw)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed as exc:
                log.warning("Gemini WS closed: %s", exc)
                break
            except Exception as exc:
                log.error("Gemini recv error: %s", exc, exc_info=True)
                break

    async def _handle_server_msg(self, raw: str | bytes):
        try:
            msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
        except Exception:
            log.warning("Gemini sent non-JSON: %s", str(raw)[:200])
            return

        # ── 1. Tool call (top-level field) ──────────────────────────────────
        #    Gemini Live sends {"toolCall": {"functionCalls": [...]}} at the
        #    top level — NOT nested inside serverContent.modelTurn.parts.
        tool_call = msg.get("toolCall")
        if tool_call:
            for fc in tool_call.get("functionCalls", []):
                if fc.get("name") == "flag_scam_alert":
                    args = fc.get("args", {})
                    fc_id = fc.get("id", "")
                    log.warning(
                        "SCAM DETECTED: category=%s confidence=%s reason=%s  (id=%s)",
                        args.get("category"), args.get("confidence"),
                        args.get("reason"), fc_id,
                    )
                    await broadcast_alert({
                        "type":       "scam_alert",
                        "category":   args.get("category", "other"),
                        "confidence": args.get("confidence", 1.0),
                        "reason":     args.get("reason", ""),
                    })
                    await self._send_tool_response(fc_id, fc.get("name"), {"status": "alerted"})
                else:
                    log.info("Unknown tool call: %s args=%s", fc.get("name"), fc.get("args"))
                    await self._send_tool_response(fc.get("id", ""), fc.get("name", ""), {"status": "ok"})
            return

        # ── 2. Server content (text / audio / model turn) ───────────────────
        server_content = msg.get("serverContent", {})
        if server_content:
            model_turn = server_content.get("modelTurn", {})
            parts = model_turn.get("parts", [])

            # Alternate format: function calls INSIDE modelTurn parts
            for part in parts:
                fc = part.get("functionCall")
                if fc and fc.get("name") == "flag_scam_alert":
                    args = fc.get("args", {})
                    fc_id = fc.get("id", "")
                    log.warning("SCAM DETECTED (in modelTurn): %s", args)
                    await broadcast_alert({
                        "type":       "scam_alert",
                        "category":   args.get("category", "other"),
                        "confidence": args.get("confidence", 1.0),
                        "reason":     args.get("reason", ""),
                    })
                    await self._send_tool_response(fc_id, fc.get("name"), {"status": "alerted"})

            text_parts = [p.get("text", "") for p in parts if "text" in p]
            if text_parts:
                log.info("Gemini text: %s", " ".join(text_parts))

            if server_content.get("turnComplete"):
                log.debug("Gemini turn complete.")

        if "setupComplete" in msg:
            log.info("Gemini setupComplete (late).")

        if not tool_call and not server_content and "setupComplete" not in msg:
            log.debug("Gemini unclassified msg: %s", json.dumps(msg)[:300])

    async def _send_tool_response(self, fc_id: str, fc_name: str, result: dict):
        """Send a tool response back to Gemini — required after function calls."""
        tool_response = {
            "toolResponse": {
                "functionResponses": [{
                    "id":       fc_id,
                    "name":     fc_name,
                    "response": result,
                }]
            }
        }
        try:
            resp_json = json.dumps(tool_response)
            log.info("Sending tool response: %s", resp_json[:300])
            await self._ws.send(resp_json)
        except Exception as exc:
            log.error("Failed to send tool response: %s", exc)


# One active session per CallUUID
active_sessions: dict[str, GeminiSession] = {}

# ─── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("CriNet backend starting — PUBLIC_HOST=%s", PUBLIC_HOST)
    if not VOBIZ_AUTH_ID or not VOBIZ_AUTH_TOKEN:
        log.error("⛔  VOBIZ_AUTH_ID / VOBIZ_AUTH_TOKEN not set in .env!")
    if not GEMINI_API_KEY:
        log.error("⛔  GEMINI_API_KEY not set in .env!")
    if not PUBLIC_HOST or "YOUR_NGROK" in PUBLIC_HOST:
        log.warning("⚠️  PUBLIC_HOST not configured — set ngrok URL in .env first")
    yield
    log.info("CriNet backend shutting down.")

app = FastAPI(title="CriNet Scam Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/", response_class=PlainTextResponse)
async def health():
    return "CriNet OK"

# ─── Serve phone alert UI ─────────────────────────────────────────────────────
@app.get("/app", response_class=FileResponse)
async def serve_alert_app():
    """Open this URL on the phone's Chrome browser during the demo."""
    html_path = Path(__file__).parent / "alert.html"
    return FileResponse(str(html_path), media_type="text/html")

# ─── Phase 1: Trigger outbound call ──────────────────────────────────────────
@app.post("/call/start")
async def start_call(request: Request):
    """
    Body (JSON):
      { "to": "+91XXXXXXXXXX" }          — number to call (the "scammer")
      { "to": "...", "victim": "+91..." } — optionally bridge to victim too
    """
    body = await request.json()
    to_number = body.get("to")
    if not to_number:
        return JSONResponse({"error": "Missing 'to' field"}, status_code=400)

    answer_url = f"{PUBLIC_HOST}/answer"
    payload = {
        "from":       VOBIZ_PHONE,
        "to":         to_number,
        "answer_url": answer_url,
    }

    # Optionally stash victim number for use in /answer
    victim_num = body.get("victim")
    if victim_num:
        # Store under MANY key variants so /answer matches regardless of how
        # Vobiz formats the To/From fields (with +, without +, full digits, last 10)
        # Also store under the VOBIZ_PHONE number since for outbound calls,
        # /answer may receive From=VOBIZ_PHONE and To=scammer.
        digits_fn = lambda s: "".join(c for c in s if c.isdigit())
        full_digits = digits_fn(to_number)          # e.g. 919959688094
        last10      = full_digits[-10:]              # e.g.   9959688094
        vobiz_digits = digits_fn(VOBIZ_PHONE)       # e.g. 918071583332
        vobiz_last10 = vobiz_digits[-10:]            # e.g.   8071583332
        keys_to_store = [
            to_number,                  # +919959688094
            to_number.lstrip("+"),      # 919959688094
            full_digits,                # 919959688094 (same as above for most)
            last10,                     # 9959688094
            VOBIZ_PHONE,                # +918071583332 (From in /answer)
            VOBIZ_PHONE.lstrip("+"),    # 918071583332
            vobiz_digits,               # 918071583332
            vobiz_last10,               # 8071583332
        ]
        for key in keys_to_store:
            pending_victims[key] = victim_num
        log.info(
            "Stored victim %s for scammer %s under %d keys: %s",
            victim_num, to_number, len(keys_to_store), keys_to_store,
        )

    headers = {
        "X-Auth-ID":    VOBIZ_AUTH_ID,
        "X-Auth-Token": VOBIZ_AUTH_TOKEN,
        "Content-Type": "application/json",
    }
    # Step 1: Update the Vobiz Application's answer_url to our current ngrok URL
    app_id = "16055179134396672"
    app_update_url = f"{VOBIZ_API_BASE}/Account/{VOBIZ_AUTH_ID}/Application/{app_id}/"
    async with httpx.AsyncClient(timeout=15) as client:
        upd = await client.post(
            app_update_url,
            json={"answer_url": answer_url, "hangup_url": f"{PUBLIC_HOST}/stream-status"},
            headers=headers,
        )
    log.info("App answer_url update: %d %s", upd.status_code, upd.text[:200])

    # Step 2: Place the outbound call
    call_url = f"{VOBIZ_API_BASE}/Account/{VOBIZ_AUTH_ID}/Call/"
    log.info("Placing outbound call: %s → %s  answer_url=%s", VOBIZ_PHONE, to_number, answer_url)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(call_url, json=payload, headers=headers)

    log.info("Vobiz response %d: %s", resp.status_code, resp.text[:300])

    if resp.status_code not in (200, 201, 202):
        log.error("⛔ Vobiz API error: %s %s", resp.status_code, resp.text)
        return JSONResponse(
            {"error": "Vobiz API error", "detail": resp.text},
            status_code=502,
        )

    return JSONResponse({"status": "call_placed", "vobiz": resp.json()})


# ─── Phase 1: Answer webhook (returns XML to Vobiz) ──────────────────────────
@app.post("/answer")
async def answer_webhook(request: Request):
    form = await request.form()
    all_fields = dict(form)
    log.info("═══ /answer webhook ALL fields: %s", all_fields)

    call_uuid = form.get("CallUUID", "unknown")
    from_num  = form.get("From", "")
    to_num    = form.get("To", "")
    direction = form.get("Direction", "")
    log.info(
        "/answer → CallUUID=%s  From=%s  To=%s  Direction=%s",
        call_uuid, from_num, to_num, direction,
    )

    # ── Build every possible lookup key from both To and From ─────────────
    digits = lambda s: "".join(c for c in s if c.isdigit())
    lookup_keys = []
    for raw in (to_num, from_num):
        if not raw:
            continue
        raw_stripped = raw.lstrip("+")
        raw_digits   = digits(raw)
        raw_last10   = raw_digits[-10:] if len(raw_digits) >= 10 else raw_digits
        lookup_keys.extend([raw, raw_stripped, raw_digits, raw_last10])

    log.info(
        "Victim lookup — lookup_keys=%s  stored_keys=%s  stored_values=%s",
        lookup_keys,
        list(pending_victims.keys()),
        list(set(pending_victims.values())),
    )

    # Try every lookup key until we find a match
    victim = ""
    matched_key = None
    for key in lookup_keys:
        v = pending_victims.get(key)
        if v:
            victim = v
            matched_key = key
            break

    # Last-resort: if exactly one victim is pending and lookup still failed, use it
    if not victim and len(set(pending_victims.values())) == 1:
        victim = next(iter(pending_victims.values()))
        matched_key = "FALLBACK-single-pending"
        log.warning("Victim lookup fell back to single-pending-victim: %s", victim)

    if victim:
        log.info("✅ Victim found: %s  (matched on key=%r)", victim, matched_key)
    else:
        log.warning("⚠️ No victim found for this call — will use single-number stream flow")

    ws_url       = f"{PUBLIC_HOST.replace('https://', 'wss://')}/media-stream"
    status_url   = f"{PUBLIC_HOST}/stream-status"
    dial_url     = f"{PUBLIC_HOST}/dial-result"
    callback_url = f"{PUBLIC_HOST}/dial-callback"

    if victim:
        # Two-number flow: <Dial redirect="false"> bridges scammer ↔ victim
        # AND <Stream keepCallAlive="true"> forks audio for Gemini monitoring.
        # With redirect="false", Vobiz moves to <Stream> immediately after
        # starting the Dial in the background — both run simultaneously.
        active_call_uuids[call_uuid] = victim
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial timeout="30"
        action="{dial_url}"
        callbackUrl="{callback_url}"
        callerId="{VOBIZ_PHONE}"
        redirect="false">
    <Number>{victim}</Number>
  </Dial>
  <Stream bidirectional="true"
          audioTrack="inbound"
          contentType="audio/x-l16;rate=16000"
          keepCallAlive="true"
          statusCallbackUrl="{status_url}">
    {ws_url}
  </Stream>
</Response>"""
        log.info(
            "Two-number flow XML: Dial+Stream → victim %s  (uuid=%s)",
            victim, call_uuid,
        )
    else:
        # Single-number flow: keep call alive and stream audio for monitoring
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true"
          audioTrack="inbound"
          contentType="audio/x-l16;rate=16000"
          keepCallAlive="true"
          statusCallbackUrl="{status_url}">
    {ws_url}
  </Stream>
</Response>"""
        log.info("Single-number flow (no victim): streaming audio for %s", call_uuid)

    log.info("Returning XML:\n%s", xml)
    return PlainTextResponse(content=xml, media_type="application/xml")


# ─── Phase 1: Stream status callback ─────────────────────────────────────────
@app.post("/stream-status")
async def stream_status(request: Request):
    form = await request.form()
    log.info("Stream status: %s", dict(form))
    return PlainTextResponse("OK")


# ─── Phase 1: Dial action URL (fires AFTER dial ends, redirect="false"→skipped) ──
@app.post("/dial-result")
async def dial_result(request: Request):
    """
    Vobiz action URL — fires once after <Dial> completes.
    With redirect="false" in the XML, this is just for logging.
    If the stream somehow wasn't started yet, return <Stream> XML as fallback.
    """
    form = await request.form()
    log.info("Dial result ALL fields: %s", dict(form))

    dial_status  = form.get("DialStatus", "")
    a_leg_uuid   = form.get("DialALegUUID", "") or form.get("CallUUID", "")

    log.info("DialStatus=%s  a_leg=%s", dial_status, a_leg_uuid)

    # Safety net: if stream wasn't started during the call, start it now
    ws_url     = f"{PUBLIC_HOST.replace('https://', 'wss://')}/media-stream"
    status_url = f"{PUBLIC_HOST}/stream-status"
    stream_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true"
          audioTrack="inbound"
          contentType="audio/x-l16;rate=16000"
          keepCallAlive="true"
          statusCallbackUrl="{status_url}">
    {ws_url}
  </Stream>
</Response>"""
    log.info("Returning Stream XML fallback after dial (DialStatus=%s)", dial_status)
    return PlainTextResponse(content=stream_xml, media_type="application/xml")


# ─── Phase 2: Dial callbackUrl handler (fires LIVE during dial) ───────────────
@app.post("/dial-callback")
async def dial_callback(request: Request):
    """
    Vobiz callbackUrl — fires live events during <Dial>:
      DialAction = answer/connected → victim picked up / bridged
      DialAction = hangup → victim hung up
    As a BACKUP, we try starting the audio stream via REST API.
    The primary path is the XML <Stream> in /answer.
    """
    form = await request.form()
    log.info("Dial callbackUrl ALL fields: %s", dict(form))

    dial_action = form.get("DialAction", "")
    a_leg_uuid  = (
        form.get("DialALegUUID", "")
        or form.get("CallUUID", "")
        or form.get("ALegUUID", "")
    )
    b_leg_uuid  = form.get("DialBLegUUID", "")
    log.info("DialAction=%s  a_leg=%s  b_leg=%s", dial_action, a_leg_uuid, b_leg_uuid)

    # BACKUP: try REST stream start when victim answers/bridges
    # The XML <Stream> in /answer is the primary path; this is insurance.
    if dial_action in ("answer", "connected") and a_leg_uuid:
        if a_leg_uuid in active_sessions or b_leg_uuid in active_sessions:
            log.info("Session already active, skipping REST start")
            return PlainTextResponse("OK")

        ws_url     = f"{PUBLIC_HOST.replace('https://', 'wss://')}/media-stream"
        status_url = f"{PUBLIC_HOST}/stream-status"
        headers = {
            "X-Auth-ID":    VOBIZ_AUTH_ID,
            "X-Auth-Token": VOBIZ_AUTH_TOKEN,
            "Content-Type": "application/json",
        }
        payload = {
            "service_url":         ws_url,
            "bidirectional":       True,
            "audio_track":         "inbound",
            "content_type":        "audio/x-l16;rate=16000",
            "status_callback_url": status_url,
        }

        # Try A-leg UUID with retries (404 = call not yet live, retry after delay)
        stream_url = f"{VOBIZ_API_BASE}/Account/{VOBIZ_AUTH_ID}/Call/{a_leg_uuid}/Stream/"
        async with httpx.AsyncClient(timeout=15) as client:
            for attempt in range(1, 4):
                log.info("REST stream start attempt %d/3 on a_leg=%s …", attempt, a_leg_uuid)
                resp = await client.post(stream_url, json=payload, headers=headers)
                log.info("  → %d %s", resp.status_code, resp.text[:200])
                if resp.status_code in (200, 201, 202):
                    log.info("✅ REST stream started successfully on a_leg=%s", a_leg_uuid)
                    break
                if resp.status_code == 404:
                    await asyncio.sleep(attempt)
                else:
                    log.warning("Non-404 REST stream error: %s", resp.text[:200])
                    break

    return PlainTextResponse("OK")


# ─── Phase 2/3: Vobiz media-stream WebSocket ─────────────────────────────────
@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("══════ Vobiz media-stream WebSocket CONNECTED ══════")

    call_uuid: Optional[str] = None
    session:   Optional[GeminiSession] = None

    try:
        async for raw in ws.iter_text():
            msg = json.loads(raw)
            event = msg.get("event")

            # ── start ────────────────────────────────────────────────────────
            if event == "start":
                start_data = msg.get("start", {})
                call_uuid  = start_data.get("callId", "unknown")
                stream_id  = start_data.get("streamId", "")
                log.info(
                    "Stream START — callId=%s streamId=%s format=%s",
                    call_uuid, stream_id,
                    start_data.get("mediaFormat"),
                )
                # Always create a new session for this stream
                if call_uuid in active_sessions:
                    session = active_sessions[call_uuid]
                    log.info("Reusing existing Gemini session for callId=%s", call_uuid)
                else:
                    session = GeminiSession()
                    active_sessions[call_uuid] = session
                    asyncio.create_task(session.start())
                    log.info("New Gemini session created for callId=%s", call_uuid)

            # ── media ────────────────────────────────────────────────────────
            elif event == "media" and session:
                media   = msg.get("media", {})
                payload = media.get("payload", "")
                if not payload:
                    continue

                # base64-decode → big-endian PCM16 → swap to little-endian
                be_bytes = base64.b64decode(payload)
                n_samples = len(be_bytes) // 2
                if n_samples > 0:
                    le_bytes = struct.pack(
                        f"<{n_samples}h",
                        *struct.unpack(f">{n_samples}h", be_bytes[:n_samples * 2])
                    )
                    await session.push_audio(le_bytes)

            # ── anything else ────────────────────────────────────────────────
            else:
                if event not in ("media",):
                    log.info("WS event: %s", event)

    except WebSocketDisconnect:
        log.info("Vobiz WS disconnected (callId=%s)", call_uuid)
    except Exception as exc:
        log.error("media-stream error: %s", exc, exc_info=True)
    finally:
        if session:
            await session.stop()
        if call_uuid and call_uuid in active_sessions:
            del active_sessions[call_uuid]
        log.info("Session cleaned up for callId=%s", call_uuid)


# ─── Phase 4: Alert push to phone "app" ──────────────────────────────────────
@app.websocket("/alerts")
async def alerts_ws(ws: WebSocket):
    await ws.accept()
    alert_subscribers.append(ws)
    log.info("Alert subscriber connected (total=%d)", len(alert_subscribers))
    try:
        # Keep alive — wait until client disconnects
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in alert_subscribers:
            alert_subscribers.remove(ws)
        log.info("Alert subscriber disconnected (total=%d)", len(alert_subscribers))


# ─── Manual alert trigger (for testing without a real call) ──────────────────
@app.post("/test-alert")
async def test_alert(request: Request):
    body = await request.json()
    payload = {
        "type":       "scam_alert",
        "category":   body.get("category", "otp_request"),
        "confidence": body.get("confidence", 0.97),
        "reason":     body.get("reason", "Test alert triggered manually"),
    }
    await broadcast_alert(payload)
    return JSONResponse({"status": "broadcast", "payload": payload})


# ─── Debug endpoint: inspect pending state ────────────────────────────────────
@app.get("/debug/pending")
async def debug_pending():
    """Shows the current pending_victims and active_call_uuids mappings for debugging."""
    return JSONResponse({
        "pending_victims": pending_victims,
        "active_call_uuids": active_call_uuids,
        "active_sessions": list(active_sessions.keys()),
        "alert_subscribers": len(alert_subscribers),
    })
