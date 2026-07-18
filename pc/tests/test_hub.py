"""Hub layer tests: registry, policy, proactive, sessions, REST + WS gateway,
MCP server. All offline (hash embedder, in-memory DB)."""
import json
import os
import time

import pytest

os.environ["RECALL_DB_PATH"] = ":memory:"
os.environ["RECALL_BACKEND"] = "hash"
# Hermetic regardless of the developer's local pc/.env: cloud_optin=True there
# would otherwise make flush_audio's now-primary Sarvam path attempt REAL
# network calls in tests that don't explicitly mock sarvam_cloud (every test
# exercising WS audio ingest does, since Sarvam is tried first when allowed).
os.environ["RECALL_CLOUD_OPTIN"] = "false"

import anyio
from fastapi.testclient import TestClient

import hub.app as hub_app
from hub.asr import (FasterWhisperProvider, PassthroughProvider,
                     WhisperLocalProvider, make_local_whisper, select_provider,
                     transcribe_with_fallback)
from hub.mcp_server import build_server
from hub.policy import PolicyEngine
from hub.proactive import ProactiveRecallEngine
from hub.registry import DeviceRegistry
from recall_memory.config import RecallConfig
from recall_memory.demo_data import seed
from recall_memory.ingest import Ingestor
from recall_memory.retrieval import Retriever
from recall_memory.store import MemoryStore


@pytest.fixture()
def state(monkeypatch):
    """Fresh HubState wired to :memory: + hash embedder (no Ollama)."""
    cfg = RecallConfig(db_path=":memory:", backend="hash")
    st = hub_app.HubState(cfg)
    monkeypatch.setattr(hub_app, "STATE", st)
    return st


@pytest.fixture()
def client(state):
    with TestClient(hub_app.app) as c:
        yield c


# ------------------------------------------------------------- registry

def test_registry_resume_and_lease():
    reg = DeviceRegistry()
    dev, resumed = reg.hello("phone-1", "phone")
    assert not resumed
    dev2, resumed2 = reg.hello("phone-1", "phone", resume_token=dev.resume_token)
    assert resumed2 and dev2.resume_token == dev.resume_token
    dev2.last_heartbeat = time.time() - 60
    assert reg.sweep() == ["phone-1"]
    assert not reg.devices["phone-1"].connected


# --------------------------------------------------------------- policy

def test_policy_gates_cloud():
    p = PolicyEngine()
    assert not p.is_cloud_allowed()                       # default OFF
    assert not p.queue_cloud_job("asr", {})
    p.update(cloud_optin=True)
    assert p.is_cloud_allowed("normal")
    assert not p.is_cloud_allowed("private")              # tag still wins
    assert p.queue_cloud_job("asr", {"x": 1})
    assert p.snapshot()["outbox_pending"] == 1
    with pytest.raises(ValueError):
        p.update(default_privacy_tag="nope")


def test_asr_selection_policy():
    p = PolicyEngine()
    providers = {"passthrough": PassthroughProvider(),
                 "whisper_local": PassthroughProvider(),
                 "hinglish_local": PassthroughProvider()}
    providers["whisper_local"].name = "whisper_local"
    providers["hinglish_local"].name = "hinglish_local"
    assert select_provider("en", 0.99, p, providers) is providers["whisper_local"]
    # Indic + no cloud opt-in -> hinglish local, never cloud
    assert select_provider("hi", 0.9, p, providers) is providers["hinglish_local"]
    res = transcribe_with_fallback({"text": "namaste"}, "hi", 0.9, p, providers)
    assert res.text == "namaste"


def test_make_local_whisper_modes():
    cfg = RecallConfig(backend="hash", asr_mode="auto")
    # auto: faster-whisper is installed in this venv -> in-process provider
    assert isinstance(make_local_whisper(cfg), FasterWhisperProvider)
    # http: always the external-server contract
    cfg_http = RecallConfig(backend="hash", asr_mode="http")
    assert isinstance(make_local_whisper(cfg_http), WhisperLocalProvider)


def test_hallucination_loop_collapsed():
    from hub.asr import _collapse_hallucination_loop
    assert _collapse_hallucination_loop("Yeah. " * 23) == "Yeah."
    normal = "we decided to use JWT authentication for the backend API"
    assert _collapse_hallucination_loop(normal) == normal


# --------------------------------------------------------------- sarvam

def test_sarvam_provider_no_key_raises_without_network_call():
    from hub.asr import SarvamCloudProvider
    p = SarvamCloudProvider(api_key="")
    with pytest.raises(RuntimeError, match="RECALL_SARVAM_API_KEY"):
        p.transcribe({"audio": b"RIFF....WAVEfmt "})


def test_sarvam_provider_parses_real_rest_contract(monkeypatch):
    """docs.sarvam.ai/api-reference/speech-to-text/transcribe: POST
    api.sarvam.ai/speech-to-text, header api-subscription-key, multipart
    file/model/language_code/mode, response {transcript, language_code,
    language_probability}."""
    from hub.asr import SarvamCloudProvider
    import requests
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"transcript": "namaste duniya", "language_code": "hi-IN",
                    "language_probability": 0.93}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        captured.update(url=url, headers=headers, data=data, timeout=timeout)
        return FakeResp()
    monkeypatch.setattr(requests, "post", fake_post)
    p = SarvamCloudProvider(api_key="test-key", model="saaras:v3",
                            language_code="unknown", timeout=3.0)
    res = p.transcribe({"audio": b"RIFF....WAVEfmt "})
    assert res.text == "namaste duniya"
    assert res.lang == "hi-IN"
    assert res.confidence == pytest.approx(0.93)
    assert captured["url"] == "https://api.sarvam.ai/speech-to-text"
    assert captured["headers"]["api-subscription-key"] == "test-key"
    assert captured["data"]["model"] == "saaras:v3"
    assert captured["timeout"] == 3.0


# ----------------------------------------------------------- sarvam tts

def test_sarvam_tts_no_key_raises_without_network_call():
    from hub.tts import SarvamTTSProvider
    p = SarvamTTSProvider(api_key="")
    with pytest.raises(RuntimeError, match="RECALL_SARVAM_API_KEY"):
        p.synthesize("hello")


def test_sarvam_tts_rejects_empty_text():
    from hub.tts import SarvamTTSProvider
    p = SarvamTTSProvider(api_key="test-key")
    with pytest.raises(ValueError):
        p.synthesize("   ")


def test_sarvam_tts_parses_real_rest_contract(monkeypatch):
    """docs.sarvam.ai/api-reference/text-to-speech/convert: POST
    api.sarvam.ai/text-to-speech, header api-subscription-key, JSON body
    text/target_language_code/model/speaker, response {"audios": [b64]}."""
    from hub.tts import SarvamTTSProvider
    import base64
    import requests
    captured = {}
    raw_audio = b"RIFF....WAVEfmt fake-wav-bytes"

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"audios": [base64.b64encode(raw_audio).decode()]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return FakeResp()
    monkeypatch.setattr(requests, "post", fake_post)
    p = SarvamTTSProvider(api_key="test-key", model="bulbul:v3", speaker="anushka",
                         language_code="en-IN", timeout=15.0)
    audio = p.synthesize("Hello there")
    assert audio == raw_audio
    assert captured["url"] == "https://api.sarvam.ai/text-to-speech"
    assert captured["headers"]["api-subscription-key"] == "test-key"
    assert captured["body"]["text"] == "Hello there"
    assert captured["body"]["target_language_code"] == "en-IN"
    assert captured["body"]["model"] == "bulbul:v3"
    assert captured["timeout"] == 15.0


def test_tts_endpoint_requires_cloud_optin(client, state):
    r = client.post("/tts", json={"text": "hello"})
    assert r.status_code == 403


def test_tts_endpoint_returns_audio_when_allowed(client, state, monkeypatch):
    state.policy.update(cloud_optin=True)
    monkeypatch.setattr(state.tts, "synthesize",
                        lambda text, **kw: b"fake-audio-bytes")
    r = client.post("/tts", json={"text": "hello there"})
    assert r.status_code == 200
    assert r.content == b"fake-audio-bytes"
    assert r.headers["content-type"] == "audio/wav"


# ------------------------------------------------------------ proactive

def test_proactive_fires_once_with_cooldown():
    store = MemoryStore(":memory:")
    cfg = RecallConfig(db_path=":memory:", backend="hash")
    ing = Ingestor(store, cfg)
    seed(ing)
    eng = ProactiveRecallEngine(Retriever(store, cfg, ing.backend),
                                threshold=0.05)
    now = time.time()
    card = eng.observe("we decided to use JWT authentication with refresh "
                       "tokens for the backend API", now=now)
    assert card and card["memory_id"]
    # cooldown suppresses the immediate next hit
    assert eng.observe("JWT authentication refresh tokens backend",
                       now=now + 1) is None
    # after cooldown it can fire again
    assert eng.observe("JWT authentication refresh tokens backend API decision",
                       now=now + 120) is not None


def test_proactive_attempt_throttle_even_without_a_fire():
    """Regression: the fire-cooldown alone is fail-OPEN — it only ever moves
    on a SUCCESSFUL recall, so a dry spell (nothing in the store matches)
    never throttled anything, and every single ASR-transcribed chunk
    re-embedded + retrieved. The attempt-throttle must gate every call,
    fire or not."""
    store = MemoryStore(":memory:")           # nothing ingested -> always dry
    cfg = RecallConfig(db_path=":memory:", backend="hash")
    ing = Ingestor(store, cfg)
    eng = ProactiveRecallEngine(Retriever(store, cfg, ing.backend), threshold=0.05)
    calls = {"n": 0}
    orig_retrieve = eng.retriever.retrieve

    def counting_retrieve(*a, **kw):
        calls["n"] += 1
        return orig_retrieve(*a, **kw)
    eng.retriever.retrieve = counting_retrieve
    now = time.time()
    for i in range(10):    # 10 rapid observes across ~9 s of continuous speech
        eng.observe(f"just talking about nothing in particular {i}", now=now + i)
    assert calls["n"] <= 2, "attempt-throttle must gate dry (never-fires) calls too"


# ------------------------------------------------------------ REST layer

def test_rest_health_stats_policy(client):
    assert client.get("/health").json()["ok"] is True
    st = client.get("/stats").json()
    assert st["memories"] == 0
    out = client.post("/policy", json={"cloud_optin": True}).json()
    assert out["cloud_optin"] is True
    assert client.get("/policy").json()["cloud_optin"] is True


def test_rest_ask_and_forget_flow(client, state):
    seed(state.ingestor)
    out = client.post("/ask", json={"query": "Who decided to use JWT?"}).json()
    assert out["sources"], "ask must return sources"
    assert out["plan"]["query_type"] == "general"   # no more per-query classification
    assert all(s["chunk_title"] for s in out["sources"])
    dump = client.get("/dump").json()
    assert len(dump) == 5
    out = client.post("/forget", json={"meeting_id": "mtg-auth-sync",
                                       "last_minutes": 1e6}).json()
    assert out["episodes"] > 0
    assert client.post("/forget", json={}).status_code == 400


def test_rest_consolidate(client, state):
    seed(state.ingestor)
    out = client.post("/consolidate").json()
    assert out["dualmic_merged"] == 1


def test_led_auto_resets_to_idle_after_delay(state):
    """The dashboard LED must not stay lit "recalled"/"error" forever — it
    was found stuck on green after an /ask completed, since nothing ever
    reset it back to idle."""
    async def _run():
        await state.set_led("recalled", reset_after=0.01)
        await anyio.sleep(0.05)
        return state.led_state
    assert anyio.run(_run) == "idle"


def test_led_auto_reset_does_not_clobber_a_newer_state(state):
    async def _run():
        await state.set_led("recalled", reset_after=0.01)
        await state.set_led("capturing")   # something newer happened meanwhile
        await anyio.sleep(0.05)
        return state.led_state
    assert anyio.run(_run) == "capturing"


# -------------------------------------------------------------- WS layer

def test_ws_meeting_lifecycle(client, state):
    seed(state.ingestor)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "unoq-1",
                                 "role": "unoq"}))
        welcome = json.loads(ws.receive_text())
        assert welcome["type"] == "welcome" and welcome["resume_token"]
        json.loads(ws.receive_text())  # device_health broadcast

        ws.send_text(json.dumps({"type": "meeting_start",
                                 "title": "standup"}))
        # led -> capturing event, then meeting_started ack
        msgs = [json.loads(ws.receive_text()) for _ in range(2)]
        types = {m.get("type") for m in msgs}
        assert "meeting_started" in types
        mid = next(m["meeting_id"] for m in msgs
                   if m.get("type") == "meeting_started")

        ws.send_text(json.dumps({"type": "utterance", "speaker": "Priya",
                                 "text": "we decided to use JWT authentication "
                                         "with refresh tokens for the API"}))
        evt = json.loads(ws.receive_text())
        while evt.get("topic") not in ("transcript",):
            evt = json.loads(ws.receive_text())
        assert evt["data"]["speaker"] == "Priya"

        ws.send_text(json.dumps({"type": "button", "action": "bookmark"}))
        msg = json.loads(ws.receive_text())
        while msg.get("type") != "ack":
            msg = json.loads(ws.receive_text())
        assert msg["ok"] is True

        ws.send_text(json.dumps({"type": "meeting_end"}))
        msg = json.loads(ws.receive_text())
        while msg.get("type") != "meeting_ended":
            msg = json.loads(ws.receive_text())
        assert msg["memory_id"]

    row = state.store.get_memory(msg["memory_id"])
    assert row["importance"] == 1.0            # bookmark applied
    assert "bookmarked" in row["tags"]
    assert json.loads(row["source_meta"])["meeting_id"] == mid


def test_ws_forget_button(client, state):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "unoq-2",
                                 "role": "unoq"}))
        json.loads(ws.receive_text())
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "meeting_start"}))
        [json.loads(ws.receive_text()) for _ in range(2)]
        ws.send_text(json.dumps({"type": "utterance", "speaker": "X",
                                 "text": "something embarrassing was said"}))
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "button", "action": "forget_last",
                                 "minutes": 5}))
        msg = json.loads(ws.receive_text())
        while msg.get("type") != "ack":
            msg = json.loads(ws.receive_text())
        assert msg["live_utterances"] == 1
        ws.send_text(json.dumps({"type": "meeting_end"}))
        msg = json.loads(ws.receive_text())
        while msg.get("type") != "meeting_ended":
            msg = json.loads(ws.receive_text())
        assert msg["memory_id"] is None        # nothing left to ingest


def test_ws_duplicate_device_id_both_receive_broadcasts(client, state):
    """Two tabs helloing as the same device_id must BOTH keep receiving
    events — a later connection must not steal the earlier one's stream."""
    with client.websocket_connect("/ws") as ws1, \
         client.websocket_connect("/ws") as ws2:
        for ws in (ws1, ws2):
            ws.send_text(json.dumps({"type": "hello", "device_id": "dashboard",
                                     "role": "dashboard"}))
        # drain welcomes + device_health broadcasts
        for ws in (ws1, ws2):
            while json.loads(ws.receive_text()).get("type") != "welcome":
                pass
        # a third connection triggers a broadcast both must see
        with client.websocket_connect("/ws") as ws3:
            ws3.send_text(json.dumps({"type": "hello", "device_id": "unoq-9",
                                      "role": "unoq"}))
            def saw_unoq9(ws):
                # earlier device_health broadcasts (from the dashboards' own
                # hellos) queue first — read until unoq-9 shows up
                for _ in range(10):
                    m = json.loads(ws.receive_text())
                    if m.get("topic") == "device_health" and any(
                            d["device_id"] == "unoq-9"
                            for d in m["data"]["devices"]):
                        return True
                return False
            assert saw_unoq9(ws1) and saw_unoq9(ws2)
    assert state.sockets == {}          # all connections cleaned up


def test_ws_audio_ingest(client, state, monkeypatch):
    """Binary PCM16 frames buffer, flush at ~6 s, and become utterances via
    the ASR worker (mocked whisper provider)."""
    from hub.asr import ASRResult

    def fake_transcribe(payload):
        assert payload["audio"][:4] == b"RIFF"       # WAV-wrapped
        return ASRResult(text="hello from audio", confidence=0.91)
    monkeypatch.setattr(state.asr_providers["whisper_local"], "transcribe",
                        fake_transcribe)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "web-capture",
                                 "role": "phone"}))
        json.loads(ws.receive_text())
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "meeting_start",
                                 "capture_device": "mobile"}))
        [json.loads(ws.receive_text()) for _ in range(2)]
        # 6 s of 16 kHz PCM16 silence = 192000 bytes -> triggers a flush
        ws.send_bytes(b"\x00\x00" * (16000 * 6))
        evt = json.loads(ws.receive_text())
        while evt.get("topic") != "transcript":
            evt = json.loads(ws.receive_text())
        assert evt["data"]["text"] == "hello from audio"
        assert evt["data"]["asr_provider"] == "whisper_local"
        ws.send_text(json.dumps({"type": "meeting_end"}))
        msg = json.loads(ws.receive_text())
        while msg.get("type") != "meeting_ended":
            msg = json.loads(ws.receive_text())
        assert msg["memory_id"]
    sess_mem = state.store.get_memory(msg["memory_id"])
    assert "hello from audio" in sess_mem["content"]


def test_flush_audio_uses_sarvam_as_primary_when_cloud_allowed(client, state,
                                                               monkeypatch):
    """Sarvam is now the PRIMARY STT whenever cloud is opted in — no local
    Whisper call, no language pre-detection needed first."""
    from hub.asr import ASRResult

    def must_not_be_called(payload):
        raise AssertionError("whisper_local must not run when Sarvam succeeds")
    monkeypatch.setattr(state.asr_providers["whisper_local"], "transcribe",
                        must_not_be_called)
    monkeypatch.setattr(
        state.asr_providers["sarvam_cloud"], "transcribe",
        lambda payload: ASRResult(text="hello from sarvam", lang="en-IN",
                                  confidence=0.95))
    state.policy.update(cloud_optin=True)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "sarvam-cap",
                                 "role": "phone"}))
        json.loads(ws.receive_text()); json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "meeting_start",
                                 "capture_device": "mobile"}))
        [json.loads(ws.receive_text()) for _ in range(2)]
        ws.send_bytes(b"\x00\x00" * (16000 * 4))     # ~4 s -> triggers a flush
        evt = json.loads(ws.receive_text())
        while evt.get("topic") != "transcript":
            evt = json.loads(ws.receive_text())
        assert evt["data"]["text"] == "hello from sarvam"
        ws.send_text(json.dumps({"type": "meeting_end"}))
        while json.loads(ws.receive_text()).get("type") != "meeting_ended":
            pass


def test_flush_audio_falls_back_to_whisper_when_cloud_not_opted_in(client, state,
                                                                   monkeypatch):
    from hub.asr import ASRResult
    monkeypatch.setattr(
        state.asr_providers["whisper_local"], "transcribe",
        lambda payload: ASRResult(text="hello from whisper", lang="en",
                                  confidence=0.9))

    def must_not_be_called(payload):
        raise AssertionError("sarvam_cloud must not run without cloud opt-in")
    monkeypatch.setattr(state.asr_providers["sarvam_cloud"], "transcribe",
                        must_not_be_called)
    # cloud_optin left at its default (False)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "no-cloud-cap",
                                 "role": "phone"}))
        json.loads(ws.receive_text()); json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "meeting_start",
                                 "capture_device": "mobile"}))
        [json.loads(ws.receive_text()) for _ in range(2)]
        ws.send_bytes(b"\x00\x00" * (16000 * 4))
        evt = json.loads(ws.receive_text())
        while evt.get("topic") != "transcript":
            evt = json.loads(ws.receive_text())
        assert evt["data"]["text"] == "hello from whisper"
        ws.send_text(json.dumps({"type": "meeting_end"}))
        while json.loads(ws.receive_text()).get("type") != "meeting_ended":
            pass


def test_flush_audio_falls_back_to_whisper_on_sarvam_failure(client, state,
                                                              monkeypatch):
    """Cloud is opted in but Sarvam errors (network/timeout) — local Whisper
    must still produce a transcript rather than silently dropping the audio."""
    from hub.asr import ASRResult

    def boom(payload):
        raise RuntimeError("simulated network failure")
    monkeypatch.setattr(state.asr_providers["sarvam_cloud"], "transcribe", boom)
    monkeypatch.setattr(
        state.asr_providers["whisper_local"], "transcribe",
        lambda payload: ASRResult(text="whisper saved the day", lang="en",
                                  confidence=0.9))
    state.policy.update(cloud_optin=True)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "sarvam-fail-cap",
                                 "role": "phone"}))
        json.loads(ws.receive_text()); json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "meeting_start",
                                 "capture_device": "mobile"}))
        [json.loads(ws.receive_text()) for _ in range(2)]
        ws.send_bytes(b"\x00\x00" * (16000 * 4))
        evt = json.loads(ws.receive_text())
        while evt.get("topic") != "transcript":
            evt = json.loads(ws.receive_text())
        assert evt["data"]["text"] == "whisper saved the day"
        ws.send_text(json.dumps({"type": "meeting_end"}))
        while json.loads(ws.receive_text()).get("type") != "meeting_ended":
            pass


def test_desktop_capture_source(state):
    """A desktop capture must land as NMO source/device 'desktop' (§3/§4a),
    not the arduino default."""
    sess = state.sessions.start("desktop-capture", "desktop")
    sess.add_utterance("testing the desktop capture flow end to end")
    mid = state.sessions.end("desktop-capture")
    row = state.store.get_memory(mid)
    assert row["source"] == "desktop"
    assert row["device"] == "desktop"


def test_untitled_session_auto_titles_from_content(state):
    """No explicit title -> memory is named after what was said, not a
    generic 'Web fallback capture' label."""
    sess = state.sessions.start("web-capture", "mobile")     # no title
    sess.add_utterance("we decided to use JWT authentication with refresh "
                       "tokens for the backend API", speaker="Priya")
    mid = state.sessions.end("web-capture")
    title = state.store.get_memory(mid)["title"]
    assert title.startswith("we decided to use JWT")
    assert len(title) <= 60
    # explicit titles are kept verbatim
    sess = state.sessions.start("web-capture", "mobile", title="standup")
    sess.add_utterance("some words were said here today")
    mid = state.sessions.end("web-capture")
    assert state.store.get_memory(mid)["title"] == "standup"


def test_ws_events_legacy_alias(client, state):
    """Old 1.0 dashboards connect to /ws/events — must work, not 403."""
    with client.websocket_connect("/ws/events") as ws:
        ws.send_text(json.dumps({"type": "hello", "device_id": "old-dash",
                                 "role": "dashboard"}))
        welcome = json.loads(ws.receive_text())
        assert welcome["type"] == "welcome" and welcome["resume_token"]


def test_links_and_digest(client, state):
    from unittest.mock import patch
    import time
    import recall_memory.demo_data as dd
    dd._NOW = None
    # Force mock time to 12:00 PM local time to avoid midnight roll-over issues
    fixed_time = time.mktime((2026, 7, 19, 12, 0, 0, 6, 200, 1))
    with patch("time.time", return_value=fixed_time), \
         patch("time.localtime", return_value=time.struct_time((2026, 7, 19, 12, 0, 0, 6, 200, 1))):
        seed(state.ingestor)
        edges = client.get("/links?threshold=0.05").json()
        assert edges and {"a", "b", "score", "why"} <= set(edges[0])
        # the JWT-heavy sample set must produce at least one explained edge
        assert any(e["why"] for e in edges)
        assert client.get("/links?threshold=1.1").json() == []
        out = client.post("/digest").json()
        assert out["count"] == 5 and "Auth design sync" in out["digest"]


def test_communities_endpoint(client, state):
    seed(state.ingestor)
    assert client.get("/communities").json() == []   # dream hasn't run yet
    client.post("/consolidate")                       # MVP jobs only
    state.consolidator.job_communities()
    out = client.get("/communities").json()
    assert out and {"community_id", "label", "members"} <= set(out[0])
    assert len(out[0]["members"]) >= 2                # JWT bridges the sources


def test_memory_detail_endpoint_returns_full_content(client, state):
    """The constellation dot-click view needs the FULL captured content, not
    just the auto-generated title `/dump` returns — this is the on-demand
    detail fetch that backs it."""
    seed(state.ingestor)
    dump = client.get("/dump").json()
    mid = dump[0]["memory_id"]
    detail = client.get(f"/memory/{mid}").json()
    assert detail["memory_id"] == mid
    assert detail["content"]              # the full raw transcript/text
    assert "title" in detail and "summary" in detail
    assert isinstance(detail["chunks"], list) and detail["chunks"]
    assert client.get("/memory/does-not-exist").status_code == 404


# ------------------------------------------------------------ MCP server

@pytest.fixture()
def mcp(tmp_path):
    db = str(tmp_path / "mcp.db")
    cfg = RecallConfig(db_path=db, backend="hash")
    store = MemoryStore(db)
    seed(Ingestor(store, cfg))
    store.close()
    return build_server(db, cfg)   # (FastMCP server, RecallTools)


def test_mcp_list_tools(mcp):
    server, _tools = mcp
    names = {t.name for t in anyio.run(server.list_tools)}
    assert names == {
        "recall_search_memory", "recall_ask", "recall_bookmark_moment",
        "recall_forget_range", "recall_dump"}


def test_mcp_tool_calls(mcp):
    _server, tools = mcp
    text = tools.search_memory("JWT decision")
    assert "meeting" in text or "jwt" in text.lower()
    assert "Auth design sync" in tools.dump()
    out = tools.forget_range("mtg-auth-sync", last_minutes=1e9)
    assert "episodes" in out
