"""Hub layer tests: registry, policy, proactive, sessions, REST + WS gateway,
MCP server. All offline (hash embedder, in-memory DB)."""
import json
import os
import time

import pytest

os.environ["RECALL_DB_PATH"] = ":memory:"
os.environ["RECALL_BACKEND"] = "hash"

import anyio
from fastapi.testclient import TestClient

import hub.app as hub_app
from hub.asr import PassthroughProvider, select_provider, transcribe_with_fallback
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
    assert out["plan"]["query_type"] == "decision"
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


def test_links_and_digest(client, state):
    seed(state.ingestor)
    edges = client.get("/links?threshold=0.05").json()
    assert edges and {"a", "b", "score"} <= set(edges[0])
    assert client.get("/links?threshold=1.1").json() == []
    out = client.post("/digest").json()
    assert out["count"] == 5 and "Auth design sync" in out["digest"]


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
