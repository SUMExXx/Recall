"""PC Hub gateway — FastAPI WebSocket hub + REST API + dashboard (§4 GW).

Run:  python -m hub   (or: uvicorn hub.app:app)

WS protocol (JSON text messages, plus binary audio):
  client -> hub : hello{device_id, role, resume_token?, last_seq?}
                  heartbeat{} | subscribe{topics[]}
                  meeting_start{meeting_id?, title?, capture_device?}
                  utterance{text, speaker?, t_start?, t_end?, asr_confidence?}
                  <binary frame>  raw PCM16 mono 16 kHz audio; buffered ~4 s,
                                  transcribed via the ASR worker (in-process
                                  faster-whisper by default), then treated as
                                  an utterance
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
import logging
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
from recall_memory.tracing import ensure_trace, step

from .asr import (HinglishLocalProvider, PassthroughProvider,
                  SarvamCloudProvider, make_local_whisper)
from .metrics import Metrics
from .policy import PolicyEngine
from .proactive import ProactiveRecallEngine
from .registry import DeviceRegistry
from .sessions import SessionManager

log = logging.getLogger("recall.hub")

EVENT_BUFFER = 500
LED_STATES = ("idle", "capturing", "searching", "recalled", "muted", "error")
SAMPLE_RATE = 16000
AUDIO_FLUSH_SECONDS = 4.0        # transcribe in ~4 s windows
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
    title: str = ""          # parent memory's title
    chunk_title: str = ""    # this hit's own extractive title
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
    chunk_title: str = ""
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
        self.policy = PolicyEngine(cloud_optin=cfg.cloud_optin)
        self.metrics = Metrics()
        self.sessions = SessionManager(self.ingestor, self.store)
        self.proactive = ProactiveRecallEngine(self.retriever)
        self.asr_providers = {
            "passthrough": PassthroughProvider(),
            "whisper_local": make_local_whisper(cfg),   # in-process or HTTP
            "hinglish_local": HinglishLocalProvider(cfg.hinglish_url),
            "sarvam_cloud": SarvamCloudProvider(
                api_key=cfg.sarvam_api_key, model=cfg.sarvam_model,
                language_code=cfg.sarvam_language_code, mode=cfg.sarvam_mode,
                timeout=cfg.sarvam_timeout_s, endpoint=cfg.sarvam_endpoint),
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
        # Dream-tier scheduling: consolidate when idle, only if data changed.
        self._last_dream = time.time()
        self._dream_stats: dict | None = None

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
        log.info("utterance [%s] %s%r (conf %.2f, %d buffered)",
                 sess.meeting_id, f"{speaker}: " if speaker else "",
                 text[:80], asr_confidence, len(sess.utterances))
        await self.broadcast("transcript", {
            "meeting_id": sess.meeting_id, "speaker": u["speaker"],
            "text": u["text"], "t_start": u["t_start"],
            "asr_provider": asr_provider})

        def _observe():
            with self.metrics.timer("proactive"):
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

    async def flush_audio(self, device_id: str, min_bytes: int = 1,
                         privacy_tag: str | None = None):
        """Transcribe the device's buffered PCM16 audio via the ASR workers.

        Selection (plan §2): local Whisper runs first — it's fast, free, and
        already performs language ID as part of transcription, so there's no
        need for a separate detection pass. If the DETECTED language is Indic
        and the user has opted into cloud AND this content's privacy tag
        allows it, prefer Sarvam's transcription for that segment (bounded by
        `sarvam_timeout_s`; any failure keeps the local result). Falls to
        HinglishLocal instead when cloud isn't allowed.
        """
        buf = self.audio_buffers.get(device_id)
        if not buf or len(buf) < min_bytes:
            return
        pcm = bytes(buf)
        buf.clear()
        wav = pcm16_to_wav(pcm)

        def _asr():
            # Runs in a worker thread (anyio.to_thread below) — the trace file
            # write happens off the event loop, same as everything else here.
            with ensure_trace("asr_transcribe", device_id=device_id,
                              audio_bytes=len(wav)), self.metrics.timer("asr"):
                with step("asr:whisper_local"):
                    local = self.asr_providers["whisper_local"].transcribe(
                        {"audio": wav})
                indic = local.lang in ("hi", "hi-en")
                if not indic:
                    return local
                if self.policy.is_cloud_allowed(privacy_tag):
                    with step("asr:sarvam_cloud", detected_lang=local.lang) as s:
                        try:
                            cloud = self.asr_providers["sarvam_cloud"].transcribe(
                                {"audio": wav})
                            s.detail(used=True)
                            return cloud
                        except Exception as e:
                            s.detail(used=False, error=str(e))
                with step("asr:hinglish_local", detected_lang=local.lang) as s:
                    try:
                        hin = self.asr_providers["hinglish_local"].transcribe(
                            {"audio": wav})
                        s.detail(used=True)
                        return hin
                    except Exception as e:
                        s.detail(used=False, error=str(e))
                return local   # every Indic path failed — keep the local result
        t0 = time.perf_counter()
        try:
            res = await anyio.to_thread.run_sync(_asr)
        except Exception as e:
            log.warning("ASR failed for %s: %s", device_id, e)
            await self.broadcast("transcript", {
                "meeting_id": "", "speaker": "", "text": "",
                "error": f"ASR unavailable: {e}"})
            return
        log.info("asr %s: %.1fs audio -> %r in %.0f ms (conf %.2f)",
                 device_id, len(pcm) / (SAMPLE_RATE * 2), res.text[:60],
                 (time.perf_counter() - t0) * 1000, res.confidence)
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
                with self.metrics.timer("ask_total"):
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
                             "chunk_title": c.chunk_title,
                             "memory_id": c.memory_id,
                             "source_type": c.source_type,
                             "snippet": c.text[:200], "score": round(c.score, 3)}
                            for c in out.get("contexts", [])],
            }
            log.info("ask %r -> tier %d/%s, %d sources, %d ms", query[:60],
                     plan.tier, plan.query_type, len(payload["sources"]),
                     payload["latency_ms"])
            await self.set_led("recalled" if payload["sources"] else "idle")
            return payload
        except Exception as e:
            log.exception("ask failed: %r", query[:60])
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
    """Health probes + lease sweeps (§4 REL) + the Dream-tier scheduler:
    consolidation runs every cfg.consolidate_every_s while idle, and ONLY when
    the store changed since the last run — never inline with capture."""
    s = get_state()
    while True:
        try:
            def _probe():
                try:
                    return bool(s.backend.llm.available)
                except Exception:
                    return False
            s.health["llm"] = await anyio.to_thread.run_sync(_probe)
            with s.store.lock:
                s.health["db"] = bool(s.store.db.execute("SELECT 1").fetchone())
            s.health["checked_at"] = time.time()
            expired = s.registry.sweep()
            if expired:
                log.info("device leases expired: %s", expired)
                await s.broadcast("device_health",
                                  {"devices": s.registry.snapshot()})
            await maybe_dream(s)
        except Exception:
            log.exception("supervisor iteration failed")
        await asyncio.sleep(15)


async def maybe_dream(s: HubState):
    """Run the consolidation agent if due and the data changed."""
    every = s.cfg.consolidate_every_s
    if not every or time.time() - s._last_dream < every:
        return
    if s.sessions.active:            # someone is mid-capture — stay out of the way
        return
    snap = s.store.stats()
    s._last_dream = time.time()
    if snap == s._dream_stats:
        return                       # nothing new since the last dream pass
    def _dream():
        # ensure_trace here (not around the outer async fn) keeps the trace
        # file write in this worker thread, off the event loop. run_full()'s
        # own ensure_trace("consolidate") attaches to this same trace, so one
        # block shows every Dream-tier job's timing together.
        with ensure_trace("dream_tier"), s.metrics.timer("consolidate"):
            return s.consolidator.run_full()
    t0 = time.perf_counter()
    out = await anyio.to_thread.run_sync(_dream)
    s._dream_stats = s.store.stats()
    log.info("dream tier ran in %.0f ms: %s",
             (time.perf_counter() - t0) * 1000, out)
    await s.broadcast("consolidated", out)


async def warmup():
    """Pre-load the hot path so the FIRST ask isn't seconds of cold start:
    embedder graph, tier-2 router prototypes, and the LLM weights."""
    s = get_state()

    def _warm() -> dict:
        report: dict = {}
        with ensure_trace("warmup"):
            with step("warmup:embed_query") as st:
                try:
                    t0 = time.perf_counter()
                    s.retriever.embedder.embed_query("warmup")
                    report["embed_ms"] = round((time.perf_counter() - t0) * 1000)
                    st.detail(ms=report["embed_ms"])
                    t0 = time.perf_counter()
                    s.retriever.router._prototypes()
                    report["prototypes_ms"] = round((time.perf_counter() - t0) * 1000)
                    st.detail(prototypes_ms=report["prototypes_ms"])
                except Exception as e:
                    report["embed"] = f"unavailable: {e}"
                    st.detail(error=str(e))
            with step("warmup:llm_ping") as st:
                try:
                    if s.backend.llm.available:
                        t0 = time.perf_counter()
                        s.backend.llm.generate("Reply with exactly: OK", timeout=120)
                        report["llm_ms"] = round((time.perf_counter() - t0) * 1000)
                        st.detail(ms=report["llm_ms"])
                    else:
                        report["llm"] = "unavailable"
                        st.detail(available=False)
                except Exception as e:
                    report["llm"] = f"unavailable: {e}"
                    st.detail(error=str(e))
        return report

    log.info("warmup: %s", await anyio.to_thread.run_sync(_warm))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_state()
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(supervisor_loop()),
             loop.create_task(warmup())]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()


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
        with s.metrics.timer("search_total"):
            return s.retriever.retrieve(body.query, top_k=body.k)
    ctxs = await anyio.to_thread.run_sync(_search)
    return [SearchItem(citation=c.citation(), score=round(c.score, 3),
                       source_type=c.source_type, title=c.title,
                       chunk_title=c.chunk_title,
                       snippet=c.text[:200]) for c in ctxs]


@app.post("/forget")
async def forget_ep(body: ForgetRequest):
    s = get_state()
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
        with s.metrics.timer("consolidate"):
            return s.consolidator.run_mvp()
    return await anyio.to_thread.run_sync(_run)


@app.get("/dump")
async def dump_ep():
    s = get_state()
    with s.store.lock:
        rows = s.store.db.execute(
            """SELECT memory_id, source_type, title, created_at, importance,
                      archived FROM memories ORDER BY created_at DESC""").fetchall()
    return [dict(r) for r in rows]


@app.get("/links")
async def links_ep(threshold: float = Query(0.55)):
    """Correlation edges between memories — feeds the dashboard constellation.

    score = 0.6 * embedding cosine (mean chunk vector) + 0.4 * shared-entity
    overlap, and each edge carries `why` (the entities both memories mention)
    so a line is explainable, not vague."""
    s = get_state()
    with s.store.lock:
        rows = s.store.db.execute(
            """SELECT c.memory_id, c.emb_full FROM chunks c
               JOIN memories m ON m.memory_id = c.memory_id
               WHERE m.archived = 0""").fetchall()
        ent_rows = s.store.db.execute(
            """SELECT DISTINCT em.memory_id, em.entity_id, em.entity_text
               FROM entity_mentions em JOIN memories m
                 ON m.memory_id = em.memory_id WHERE m.archived = 0""").fetchall()
    by_mem: dict[str, list] = {}
    for r in rows:
        by_mem.setdefault(r["memory_id"], []).append(
            np.frombuffer(r["emb_full"], dtype=np.float32))
    ents: dict[str, set] = {}
    ent_text: dict[str, str] = {}
    for r in ent_rows:
        ents.setdefault(r["memory_id"], set()).add(r["entity_id"])
        ent_text.setdefault(r["entity_id"], r["entity_text"])
    ids = list(by_mem)
    if len(ids) < 2:
        return []
    mat = np.stack([np.mean(by_mem[i], axis=0) for i in ids])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sim = mat @ mat.T
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            shared = ents.get(a, set()) & ents.get(b, set())
            overlap = (len(shared) / min(len(ents.get(a, set())) or 1,
                                         len(ents.get(b, set())) or 1)
                       if shared else 0.0)
            score = 0.6 * float(sim[i, j]) + 0.4 * min(1.0, overlap)
            if score >= threshold:
                why = sorted((ent_text[e] for e in shared),
                             key=str.lower)[:3]
                edges.append({"a": a, "b": b, "score": round(score, 3),
                              "why": why})
    edges.sort(key=lambda e: -e["score"])
    return edges[:200]


@app.get("/communities")
async def communities_ep():
    """Dream-tier memory groups (§12 job 5) — clusters of memories that share
    entities, labelled by the entity they share most."""
    s = get_state()
    with s.store.lock:
        rows = s.store.db.execute(
            "SELECT community_id, label, member_ids FROM communities").fetchall()
        out = []
        for r in rows:
            label_row = s.store.db.execute(
                "SELECT entity_text FROM entity_mentions WHERE entity_id=? LIMIT 1",
                (r["label"],)).fetchone()
            out.append({"community_id": r["community_id"],
                        "label": label_row["entity_text"] if label_row else r["label"],
                        "members": json.loads(r["member_ids"])})
    return out


@app.post("/digest")
async def digest_ep():
    """Briefing: what was captured today (extractive; LLM answers live in /ask)."""
    s = get_state()
    t0 = time.perf_counter()
    day_start = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    with s.store.lock:
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


_NO_CACHE = {"Cache-Control": "no-store"}   # stale dashboards caused /ws 403s


@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "dashboard", "index.html"),
                        headers=_NO_CACHE)


@app.get("/capture")
async def capture_page():
    return FileResponse(os.path.join(os.path.dirname(__file__),
                                     "dashboard", "capture.html"),
                        headers=_NO_CACHE)


# ------------------------------------------------------------------- WS

@app.websocket("/ws")
@app.websocket("/ws/events")   # legacy 1.0-dashboard path — stale tabs got 403
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
                log.info("ws hello %s (role=%s, resumed=%s)",
                         device_id, msg.get("role", "other"), resumed)
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
                log.info("meeting_start [%s] device=%s capture=%s",
                         sess.meeting_id, device_id, sess.capture_device)
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
                    out = s.sessions.forget_last(device_id, minutes)
                    s.metrics.incr("forgets")
                    await ws.send_text(json.dumps(
                        {"type": "ack", "action": "forget_last", **out}))

            elif mtype == "meeting_end":
                await s.flush_audio(device_id)

                def _end():
                    with s.metrics.timer("ingest_meeting"):
                        return s.sessions.end(device_id)
                memory_id = await anyio.to_thread.run_sync(_end)
                log.info("meeting_end device=%s -> memory %s", device_id,
                         memory_id or "(no utterances — nothing ingested)")
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
