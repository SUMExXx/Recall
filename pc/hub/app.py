"""PC Hub gateway — FastAPI WebSocket hub + REST API + dashboard (§4 GW).

Run:  python -m hub   (or: uvicorn hub.app:app)

WS protocol (JSON text messages, plus binary audio):
  client -> hub : hello{device_id, role, resume_token?, last_seq?}
                  heartbeat{} | subscribe{topics[]}
                  meeting_start{meeting_id?, title?, capture_device?}
                  utterance{text, speaker?, t_start?, t_end?, asr_confidence?}
                  <binary frame>  raw PCM16 mono 16 kHz audio; buffered and
                                  transcribed via the ASR worker, then treated
                                  as an utterance
                  button{action: bookmark|forget_last, minutes?}
                  meeting_end{} | ask{query, request_id?}
  hub -> client : welcome{resume_token, seq, missed[]} | pong
                  event{topic, seq, data}  topics: transcript, led_state,
                  recall, device_health, answer
Every event carries a hub-wide seq number; a reconnect with resume_token +
last_seq replays what was missed from the ring buffer.

The engine's models come from the active backend (RECALL_BACKEND: npu | ollama
| hash) — this file never touches Ollama/QNN directly.
"""
from __future__ import annotations

import asyncio
import io
import itertools
import json
import os
import threading
import time
import wave
from collections import deque
from contextlib import asynccontextmanager

import anyio
import numpy as np
from fastapi import (FastAPI, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from recall_memory.backends import get_backend
from recall_memory.config import RecallConfig
from recall_memory.consolidate import Consolidator
from recall_memory.ingest import Ingestor
from recall_memory.retrieval import Retriever
from recall_memory.store import MemoryStore

from .asr import (HinglishLocalProvider, PassthroughProvider,
                  SarvamCloudProvider, WhisperLocalProvider,
                  transcribe_with_fallback)
from .metrics import Metrics
from .policy import PolicyEngine
from .proactive import ProactiveRecallEngine
from .registry import DeviceRegistry
from .sessions import SessionManager

EVENT_BUFFER = 500
LED_STATES = ("idle", "capturing", "searching", "recalled", "muted", "error")
SAMPLE_RATE = 16000
AUDIO_FLUSH_SECONDS = 6.0        # transcribe in ~6 s windows
_AUDIO_FLUSH_BYTES = int(SAMPLE_RATE * 2 * AUDIO_FLUSH_SECONDS)


# ------------------------------------------------------------ API schemas

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=50)


class ForgetRequest(BaseModel):
    memory_id: str | None = None
    meeting_id: str | None = None
    last_minutes: float | None = Field(None, gt=0)


class PolicyUpdate(BaseModel):
    cloud_optin: bool | None = None
    default_privacy_tag: str | None = None


class PlanInfo(BaseModel):
    tier: int
    query_type: str
    path: str
    weight_profile: str
    filters: dict = {}


class SourceItem(BaseModel):
    citation: str
    title: str = ""
    memory_id: str = ""
    source_type: str = ""
    snippet: str = ""
    score: float = 0.0


class AskResponse(BaseModel):
    answer: str | None = None
    command: dict | None = None
    latency_ms: int = 0
    plan: PlanInfo | None = None
    sources: list[SourceItem] = []
    error: str | None = None


class SearchItem(BaseModel):
    citation: str
    score: float
    source_type: str
    title: str
    snippet: str


class PolicyState(BaseModel):
    cloud_optin: bool
    default_privacy_tag: str
    outbox_pending: int


class HealthResponse(BaseModel):
    ok: bool
    led: str
    backend: str
    llm: bool | None = None
    db: bool = True
    checked_at: float = 0.0
    devices: list[dict] = []


def pcm16_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class HubState:
    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self.lock = threading.RLock()      # serializes store access across threads
        self.store = MemoryStore(cfg.db_path, check_same_thread=False)
        self.backend = get_backend(cfg)
        self.ingestor = Ingestor(self.store, cfg, self.backend)
        self.retriever = Retriever(self.store, cfg, self.backend)
        self.consolidator = Consolidator(self.store, cfg, self.ingestor)
        self.registry = DeviceRegistry()
        self.policy = PolicyEngine()
        self.metrics = Metrics()
        self.sessions = SessionManager(self.ingestor, self.store)
        self.proactive = ProactiveRecallEngine(self.retriever)
        self.asr_providers = {
            "passthrough": PassthroughProvider(),
            "whisper_local": WhisperLocalProvider(cfg.whisper_url),
            "hinglish_local": HinglishLocalProvider(cfg.hinglish_url),
            "sarvam_cloud": SarvamCloudProvider(),
        }
        # WS fabric — keyed per CONNECTION, not per device_id: two tabs may
        # both hello as "dashboard", and a reconnect must not steal the event
        # stream from a socket that is still alive.
        self.seq = 0
        self.events: deque = deque(maxlen=EVENT_BUFFER)  # ring buffer for resume
        self._conn_counter = itertools.count(1)
        self.sockets: dict[int, dict] = {}       # conn_id -> {device_id, ws}
        self.subscriptions: dict[int, set] = {}  # conn_id -> topics
        self.audio_buffers: dict[str, bytearray] = {}    # device_id -> pcm16
        self.led_state = "idle"
        self.health = {"llm": None, "db": True, "checked_at": 0.0}

    # ------------------------------------------------------------ events

    def _next_event(self, topic: str, data: dict) -> dict:
        self.seq += 1
        evt = {"type": "event", "topic": topic, "seq": self.seq, "data": data}
        self.events.append(evt)
        return evt

    async def broadcast(self, topic: str, data: dict):
        evt = self._next_event(topic, data)
        dead = []
        for conn_id, entry in list(self.sockets.items()):
            topics = self.subscriptions.get(conn_id)
            if topics is not None and topic not in topics:
                continue
            try:
                await entry["ws"].send_text(json.dumps(evt))
            except Exception:
                dead.append(conn_id)
        for conn_id in dead:
            self.drop_connection(conn_id)

    def drop_connection(self, conn_id: int):
        entry = self.sockets.pop(conn_id, None)
        self.subscriptions.pop(conn_id, None)
        if entry is None:
            return
        device_id = entry["device_id"]
        # only mark the device offline when its LAST connection is gone
        if not any(e["device_id"] == device_id for e in self.sockets.values()):
            self.registry.disconnect(device_id)

    async def set_led(self, state: str):
        if state not in LED_STATES:
            state = "error"
        if state != self.led_state:
            self.led_state = state
            await self.broadcast("led_state", {"state": state})

    # ---------------------------------------------------------- utterance

    async def handle_utterance(self, device_id: str, text: str,
                               speaker: str = "", asr_provider: str = "passthrough",
                               asr_confidence: float = 1.0, lang: str = "en",
                               t_start: float | None = None,
                               t_end: float | None = None):
        """Shared path for text utterances and ASR-transcribed audio."""
        sess = self.sessions.get(device_id)
        if sess is None:
            sess = self.sessions.start(device_id)
            await self.set_led("capturing")
        u = sess.add_utterance(text=text, speaker=speaker, t_start=t_start,
                               t_end=t_end, asr_provider=asr_provider,
                               asr_confidence=asr_confidence, lang=lang)
        self.metrics.incr("utterances")
        await self.broadcast("transcript", {
            "meeting_id": sess.meeting_id, "speaker": u["speaker"],
            "text": u["text"], "t_start": u["t_start"],
            "asr_provider": asr_provider})

        def _observe():
            with self.lock, self.metrics.timer("proactive"):
                return self.proactive.observe(
                    text, exclude_meeting_id=sess.meeting_id)
        recall = await anyio.to_thread.run_sync(_observe)
        if recall:
            self.metrics.incr("proactive_recalls")
            recall["new_text"] = text
            await self.broadcast("recall", recall)
            await self.set_led("recalled")
            await asyncio.sleep(0)
            await self.set_led("capturing")

    async def flush_audio(self, device_id: str, min_bytes: int = 1):
        """Transcribe the device's buffered PCM16 audio via the ASR workers."""
        buf = self.audio_buffers.get(device_id)
        if not buf or len(buf) < min_bytes:
            return
        pcm = bytes(buf)
        buf.clear()
        wav = pcm16_to_wav(pcm)
        providers = {k: self.asr_providers[k]
                     for k in ("whisper_local", "hinglish_local")}

        def _asr():
            with self.metrics.timer("asr"):
                return transcribe_with_fallback(
                    {"audio": wav}, "en", 0.99, self.policy, providers)
        try:
            res = await anyio.to_thread.run_sync(_asr)
        except Exception as e:
            await self.broadcast("transcript", {
                "meeting_id": "", "speaker": "", "text": "",
                "error": f"ASR unavailable: {e}"})
            return
        if res.text.strip():
            await self.handle_utterance(
                device_id, res.text.strip(), asr_provider="whisper_local",
                asr_confidence=res.confidence, lang=res.lang)

    # -------------------------------------------------------------- ask

    async def run_ask(self, query: str, k: int = 6) -> dict:
        await self.set_led("searching")
        self.metrics.incr("asks")
        t0 = time.perf_counter()
        try:
            def _ask():
                with self.lock, self.metrics.timer("ask_total"):
                    return self.retriever.ask(query, top_k=k)
            out = await anyio.to_thread.run_sync(_ask)
            plan = out["plan"]
            payload = {
                "answer": out.get("answer"),
                "command": out.get("command"),
                "latency_ms": round((time.perf_counter() - t0) * 1000),
                "plan": {"tier": plan.tier, "query_type": plan.query_type,
                         "path": plan.path, "weight_profile": plan.weight_profile,
                         "filters": plan.filters},
                "sources": [{"citation": c.citation(), "title": c.title,
                             "memory_id": c.memory_id,
                             "source_type": c.source_type,
                             "snippet": c.text[:200], "score": round(c.score, 3)}
                            for c in out.get("contexts", [])],
            }
            await self.set_led("recalled" if payload["sources"] else "idle")
            return payload
        except Exception as e:
            await self.set_led("error")
            return {"error": str(e),
                    "latency_ms": round((time.perf_counter() - t0) * 1000)}


STATE: HubState | None = None


def get_state() -> HubState:
    global STATE
    if STATE is None:
        STATE = HubState(RecallConfig())
    return STATE


# ------------------------------------------------------------- lifespan

async def supervisor_loop():
    """Health probes + lease sweeps (§4 REL). Workers here are in-process /
    HTTP services, so 'restart' means re-probing and surfacing state."""
    s = get_state()
    while True:
        try:
            def _probe():
                try:
                    return bool(s.backend.llm.available)
                except Exception:
                    return False
            s.health["llm"] = await anyio.to_thread.run_sync(_probe)
            with s.lock:
                s.health["db"] = bool(s.store.db.execute("SELECT 1").fetchone())
            s.health["checked_at"] = time.time()
            expired = s.registry.sweep()
            if expired:
                await s.broadcast("device_health",
                                  {"devices": s.registry.snapshot()})
        except Exception:
            pass
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_state()
    task = asyncio.get_event_loop().create_task(supervisor_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Recall PC Hub", version="2.0", lifespan=lifespan)


# ------------------------------------------------------------------ REST

@app.get("/health", response_model=HealthResponse)
async def health():
    s = get_state()
    return HealthResponse(ok=True, led=s.led_state, backend=s.backend.name,
                          devices=s.registry.snapshot(), **s.health)


@app.get("/stats")
async def stats():
    s = get_state()
    with s.lock:
        return {**s.store.stats(), "backend": s.backend.name,
                "embedder": s.backend.embedder.name}


@app.get("/metrics")
async def metrics_ep():
    return get_state().metrics.summary()


@app.get("/devices")
async def devices():
    return get_state().registry.snapshot()


@app.get("/policy", response_model=PolicyState)
async def policy_get():
    return get_state().policy.snapshot()


@app.post("/policy", response_model=PolicyState)
async def policy_post(body: PolicyUpdate):
    try:
        return get_state().policy.update(body.cloud_optin, body.default_privacy_tag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
async def ask_ep(body: AskRequest):
    payload = await get_state().run_ask(body.query, body.k)
    return AskResponse.model_validate(payload)


@app.post("/search", response_model=list[SearchItem])
async def search_ep(body: SearchRequest):
    s = get_state()

    def _search():
        with s.lock, s.metrics.timer("search_total"):
            return s.retriever.retrieve(body.query, top_k=body.k)
    ctxs = await anyio.to_thread.run_sync(_search)
    return [SearchItem(citation=c.citation(), score=round(c.score, 3),
                       source_type=c.source_type, title=c.title,
                       snippet=c.text[:200]) for c in ctxs]


@app.post("/forget")
async def forget_ep(body: ForgetRequest):
    s = get_state()
    with s.lock:
        if body.memory_id:
            out = s.store.forget_memory(body.memory_id)
        elif body.meeting_id and body.last_minutes:
            t_to = time.time()
            out = s.store.forget_time_range(
                body.meeting_id, t_to - float(body.last_minutes) * 60, t_to)
        else:
            raise HTTPException(status_code=400,
                                detail="need memory_id or meeting_id+last_minutes")
    s.metrics.incr("forgets")
    await s.broadcast("memory_deleted", {"memory_id": body.memory_id, **out})
    return out


@app.post("/consolidate")
async def consolidate_ep():
    s = get_state()

    def _run():
        with s.lock, s.metrics.timer("consolidate"):
            return s.consolidator.run_mvp()
    return await anyio.to_thread.run_sync(_run)


@app.get("/dump")
async def dump_ep():
    s = get_state()
    with s.lock:
        rows = s.store.db.execute(
            """SELECT memory_id, source_type, title, created_at, importance,
                      archived FROM memories ORDER BY created_at DESC""").fetchall()
    return [dict(r) for r in rows]


@app.get("/links")
async def links_ep(threshold: float = Query(0.6)):
    """Similarity edges between memories — feeds the dashboard constellation.
    Memory vector = mean of its chunks' full embeddings."""
    s = get_state()
    with s.lock:
        rows = s.store.db.execute(
            """SELECT c.memory_id, c.emb_full FROM chunks c
               JOIN memories m ON m.memory_id = c.memory_id
               WHERE m.archived = 0""").fetchall()
    by_mem: dict[str, list] = {}
    for r in rows:
        by_mem.setdefault(r["memory_id"], []).append(
            np.frombuffer(r["emb_full"], dtype=np.float32))
    ids = list(by_mem)
    if len(ids) < 2:
        return []
    mat = np.stack([np.mean(by_mem[i], axis=0) for i in ids])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sim = mat @ mat.T
    edges = [{"a": ids[i], "b": ids[j], "score": round(float(sim[i, j]), 3)}
             for i in range(len(ids)) for j in range(i + 1, len(ids))
             if sim[i, j] >= threshold]
    edges.sort(key=lambda e: -e["score"])
    return edges[:200]


@app.post("/digest")
async def digest_ep():
    """Briefing: what was captured today (extractive; LLM answers live in /ask)."""
    s = get_state()
    t0 = time.perf_counter()
    day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    with s.lock:
        rows = s.store.db.execute(
            """SELECT title, summary, source_type FROM memories
               WHERE archived=0 AND created_at >= ? ORDER BY created_at""",
            (day_start,)).fetchall()
    if not rows:
        digest = "Nothing captured yet today."
    else:
        digest = " · ".join(f"{r['title']}: {(r['summary'] or '').strip()[:140]}"
                            for r in rows)
    return {"digest": digest, "count": len(rows),
            "latency_ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "dashboard", "index.html"))


@app.get("/capture")
async def capture_page():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "dashboard", "capture.html"))


# ------------------------------------------------------------------- WS

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    s = get_state()
    await ws.accept()
    conn_id = next(s._conn_counter)
    device_id: str | None = None
    try:
        while True:
            frame = await ws.receive()
            if frame.get("type") == "websocket.disconnect":
                break

            # ---- binary: PCM16 audio from a capture device ----
            if frame.get("bytes") is not None:
                if device_id is None:
                    continue
                buf = s.audio_buffers.setdefault(device_id, bytearray())
                buf.extend(frame["bytes"])
                if len(buf) >= _AUDIO_FLUSH_BYTES:
                    await s.flush_audio(device_id)
                continue

            msg = json.loads(frame.get("text") or "{}")
            mtype = msg.get("type")

            if mtype == "hello":
                device_id = msg["device_id"]
                dev, resumed = s.registry.hello(
                    device_id, msg.get("role", "other"), msg.get("resume_token"))
                s.sockets[conn_id] = {"device_id": device_id, "ws": ws}
                missed = []
                if resumed and msg.get("last_seq"):
                    missed = [e for e in s.events
                              if e["seq"] > int(msg["last_seq"])]
                await ws.send_text(json.dumps({
                    "type": "welcome", "resume_token": dev.resume_token,
                    "seq": s.seq, "resumed": resumed, "missed": missed,
                    "led_state": s.led_state}))
                await s.broadcast("device_health",
                                  {"devices": s.registry.snapshot()})

            elif device_id is None:
                await ws.send_text(json.dumps(
                    {"type": "error", "error": "hello first"}))

            elif mtype == "heartbeat":
                s.registry.heartbeat(device_id)
                await ws.send_text(json.dumps({"type": "pong", "seq": s.seq}))

            elif mtype == "subscribe":
                s.subscriptions[conn_id] = set(msg.get("topics") or [])

            elif mtype == "meeting_start":
                sess = s.sessions.start(
                    device_id, msg.get("capture_device", "arduino"),
                    msg.get("meeting_id"), msg.get("title"),
                    msg.get("privacy_tag", "normal"))
                s.proactive.reset()
                s.audio_buffers[device_id] = bytearray()
                await s.set_led("capturing")
                await ws.send_text(json.dumps(
                    {"type": "meeting_started", "meeting_id": sess.meeting_id}))

            elif mtype == "utterance":
                await s.handle_utterance(
                    device_id, msg["text"], speaker=msg.get("speaker", ""),
                    asr_confidence=float(msg.get("asr_confidence", 1.0)),
                    lang=msg.get("lang", "en"),
                    t_start=msg.get("t_start"), t_end=msg.get("t_end"))

            elif mtype == "button":
                action = msg.get("action")
                if action == "bookmark":
                    ok = s.sessions.bookmark(device_id)
                    await ws.send_text(json.dumps(
                        {"type": "ack", "action": "bookmark", "ok": ok}))
                elif action == "forget_last":
                    minutes = float(msg.get("minutes", 5))
                    with s.lock:
                        out = s.sessions.forget_last(device_id, minutes)
                    s.metrics.incr("forgets")
                    await ws.send_text(json.dumps(
                        {"type": "ack", "action": "forget_last", **out}))

            elif mtype == "meeting_end":
                await s.flush_audio(device_id)

                def _end():
                    with s.lock, s.metrics.timer("ingest_meeting"):
                        return s.sessions.end(device_id)
                memory_id = await anyio.to_thread.run_sync(_end)
                await s.set_led("idle")
                await ws.send_text(json.dumps(
                    {"type": "meeting_ended", "memory_id": memory_id}))
                if memory_id:
                    await s.broadcast("memory_added", {"memory_id": memory_id})

            elif mtype == "ask":
                out = await s.run_ask(msg["query"], int(msg.get("k", 6)))
                await s.broadcast("answer",
                                  {"request_id": msg.get("request_id"), **out})

            else:
                await ws.send_text(json.dumps(
                    {"type": "error", "error": f"unknown type {mtype!r}"}))

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        s.drop_connection(conn_id)
        if device_id:
            s.audio_buffers.pop(device_id, None)
