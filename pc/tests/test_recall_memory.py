"""End-to-end tests for the Recall memory engine (deterministic HashEmbedder,
no Ollama needed)."""
import json

import numpy as np
import pytest

from recall_memory.chunker import chunk_document, chunk_meeting
from recall_memory.config import CHUNK_SPECS, DIM_COARSE, DIM_FULL, RecallConfig
from recall_memory.consolidate import Consolidator
from recall_memory.demo_data import (SAMPLE_GITHUB, SAMPLE_OCR, SAMPLE_PDF,
                                     sample_meeting, sample_meeting_phone, seed)
from recall_memory.embeddings import HashEmbedder, matryoshka_coarse
from recall_memory.ingest import Ingestor
from recall_memory.nmo import NMO, Episode, new_id
from recall_memory.retrieval import Retriever
from recall_memory.router import Router
from recall_memory.store import MemoryStore
from recall_memory.tokenizer import count_tokens, tokenize


@pytest.fixture()
def cfg():
    return RecallConfig(db_path=":memory:", embedder="hash")


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def ingestor(store, cfg):
    return Ingestor(store, cfg, HashEmbedder())


@pytest.fixture()
def seeded(store, cfg, ingestor):
    ids = seed(ingestor)
    return store, cfg, ingestor, ids


# ---------------------------------------------------------------- chunking

def test_fixed_windows_size_and_overlap():
    spec = CHUNK_SPECS["text"]
    text = " ".join(f"tok{i}" for i in range(1000))
    nmo = NMO.create("text", "desktop", "t", text)
    chunks = chunk_document(nmo)
    assert all(c.token_count <= spec.size for c in chunks)
    assert all(c.token_count == spec.size for c in chunks[:-1])
    # overlap: consecutive windows share `overlap` tokens
    t0 = tokenize(chunks[0].text)
    t1 = tokenize(chunks[1].text)
    assert [t.text for t in t0[-spec.overlap:]] == [t.text for t in t1[:spec.overlap]]
    # exact char-span provenance into NMO.content
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text


def test_image_is_whole_block():
    nmo = NMO.create("image", "mobile", "wb", SAMPLE_OCR["content"],
                     SAMPLE_OCR["source_meta"])
    chunks = chunk_document(nmo)
    assert len(chunks) == 1
    assert chunks[0].ocr_confidence == 0.88


def test_meeting_chunks_carry_episode_metadata():
    m = sample_meeting()
    eps = [Episode(episode_id=new_id(), meeting_id=m["meeting_id"], memory_id="x",
                   speaker=u["speaker"], t_start=1000 + u["t_start"],
                   t_end=1000 + u["t_end"], text=u["text"])
           for u in m["utterances"]]
    nmo = NMO.create("meeting", "arduino", m["title"], "placeholder",
                     {"meeting_id": m["meeting_id"]})
    chunks = chunk_meeting(nmo, eps)
    assert chunks, "meeting produced no chunks"
    for c in chunks:
        assert c.episode_ids, "meeting chunk missing episode_ids"
        assert c.t_start is not None and c.t_end is not None
        assert c.speaker_span
        assert c.meeting_id == m["meeting_id"]


def test_speaker_attribution_never_split():
    # Long utterances force multiple windows; every window that starts inside a
    # line must not begin right after a dangling "Speaker:" label.
    eps = [Episode(episode_id=new_id(), meeting_id="m", memory_id="x",
                   speaker=f"Speaker{i}", t_start=i * 10.0, t_end=i * 10 + 9,
                   text=" ".join(f"w{i}_{j}" for j in range(120)))
           for i in range(8)]
    nmo = NMO.create("meeting", "arduino", "t", "placeholder",
                     {"meeting_id": "m"})
    for c in chunk_meeting(nmo, eps):
        first_line = c.text.split("\n", 1)[0]
        # a window starting mid-line starts at a word; one starting at a line
        # start carries the full "SpeakerN:" label — never a bare label alone
        assert not first_line.rstrip().endswith(":") or " " in first_line.strip()


# ----------------------------------------------------------- embeddings

def test_matryoshka_coarse_shape_and_range():
    emb = HashEmbedder().embed_documents(["hello world", "jwt auth tokens"])
    assert emb.shape == (2, DIM_FULL)
    coarse = matryoshka_coarse(emb)
    assert coarse.shape == (2, DIM_COARSE)
    assert coarse.dtype == np.int8
    assert np.abs(coarse).max() <= 127


# -------------------------------------------------------------- ingestion

def test_ingest_writes_all_indexes(seeded):
    store, _, _, ids = seeded
    st = store.stats()
    assert st["memories"] == 5
    assert st["episodes"] == 9          # 6 arduino + 3 phone
    assert st["chunks"] >= 5
    assert st["chunks"] == st["vec_chunks"]
    assert st["entity_mentions"] > 0
    assert st["relations"] > 0          # decisions extracted
    fts = store.db.execute("SELECT count(*) c FROM fts_chunks").fetchone()["c"]
    assert fts == st["chunks"]


def test_decision_extraction(seeded):
    store, *_ = seeded
    rels = store.db.execute(
        "SELECT * FROM relations WHERE predicate='decided'").fetchall()
    objects = " | ".join(r["object"].lower() for r in rels)
    assert "jwt" in objects
    assert "postgresql" in objects


def test_source_specific_chunk_metadata(seeded):
    store, *_ = seeded
    gh = store.db.execute(
        "SELECT * FROM chunks WHERE source_type='github_repo' LIMIT 1").fetchone()
    assert gh["repo"] == "recall-backend"
    assert gh["file_path"] == "src/auth/jwt_middleware.py"
    pdf = store.db.execute(
        "SELECT * FROM chunks WHERE source_type='pdf' LIMIT 1").fetchone()
    assert pdf["page"] == 4
    assert pdf["document_title"] == "Recall Architecture Doc"


# ---------------------------------------------------------------- router

def test_router_tier0_forget(store, cfg):
    plan = Router(store, cfg).route("forget the last 5 minutes")
    assert plan.tier == 0
    assert plan.command == {"action": "forget_last", "minutes": 5}


def test_router_tier1_decision_skips_vector(seeded):
    store, cfg, ing, _ = seeded
    plan = Router(store, cfg, ing.embedder).route("Who decided to use JWT?")
    assert plan.tier == 1
    assert plan.query_type == "decision"
    assert plan.needs["vector"] is False
    assert plan.needs["graph"] is True


def test_router_tier1_metadata_only(seeded):
    store, cfg, ing, _ = seeded
    plan = Router(store, cfg, ing.embedder).route("show me meetings from yesterday")
    assert plan.path == "fast"
    assert plan.needs == {"bm25": False, "vector": False, "graph": False,
                          "metadata_filter": True, "entity_index": False}
    assert plan.filters.get("source_type") == "meeting"
    assert plan.filters.get("date_range") is not None


def test_router_tier1_speaker_filter(seeded):
    store, cfg, ing, _ = seeded
    plan = Router(store, cfg, ing.embedder).route("What did Priya decide?")
    assert plan.filters.get("speaker") == "Priya"


def test_router_tier2_fallthrough(seeded):
    store, cfg, ing, _ = seeded
    r = Router(store, cfg, ing.embedder)
    q = "tell me about our project direction overall"
    vec = ing.embedder.embed_query(q)
    plan = r.route(q, vec)
    assert plan.tier == 2


# ------------------------------------------------------------- retrieval

def test_retrieval_finds_decision(seeded):
    store, cfg, ing, ids = seeded
    r = Retriever(store, cfg, ing.embedder)
    ctxs = r.retrieve("Who decided to use JWT authentication?")
    assert ctxs
    joined = " ".join(c.text.lower() for c in ctxs)
    assert "jwt" in joined
    top_meeting = [c for c in ctxs if c.source_type == "meeting"]
    assert top_meeting, "decision query should surface the meeting"
    assert top_meeting[0].episodes, "meeting context must carry verbatim episodes"


def test_cross_source_bridge(seeded):
    store, cfg, ing, _ = seeded
    r = Retriever(store, cfg, ing.embedder)
    ctxs = r.retrieve("JWT", top_k=10)
    types = {c.source_type for c in ctxs}
    assert {"meeting", "github_repo", "pdf"} <= types, f"got {types}"


def test_speaker_filtered_retrieval(seeded):
    store, cfg, ing, _ = seeded
    r = Retriever(store, cfg, ing.embedder)
    ctxs = r.retrieve("What did Ananya say about PostgreSQL?")
    assert ctxs
    meeting_ctxs = [c for c in ctxs if c.source_type == "meeting"]
    assert meeting_ctxs
    spans = json.loads(meeting_ctxs[0].meta.get("speaker_span", "[]"))
    assert "Ananya" in spans


# ---------------------------------------------------------------- forget

def test_forget_time_range_cascades(seeded):
    store, cfg, ing, ids = seeded
    before = store.stats()
    m = sample_meeting()
    t0 = m["start_time"] + 25   # covers Priya's decision utterance
    out = store.forget_time_range("mtg-auth-sync", t0, t0 + 13)
    assert out["episodes"] >= 1
    assert out["chunks"] >= 1
    after = store.stats()
    assert after["episodes"] < before["episodes"]
    assert after["chunks"] < before["chunks"]
    assert after["vec_chunks"] == after["chunks"]  # vec rows cascade too
    fts = store.db.execute("SELECT count(*) c FROM fts_chunks").fetchone()["c"]
    assert fts == after["chunks"]
    # history logged on the parent memory
    hist = json.loads(store.get_memory(ids["meeting_arduino"])["history"])
    assert any(h["field"] == "forget" for h in hist)


def test_forget_memory(seeded):
    store, _, _, ids = seeded
    store.forget_memory(ids["github"])
    assert store.get_memory(ids["github"]) is None
    left = store.db.execute(
        "SELECT count(*) c FROM chunks WHERE source_type='github_repo'"
    ).fetchone()["c"]
    assert left == 0


# ------------------------------------------------------------ consolidation

def test_dedup_exact(store, cfg, ingestor):
    ingestor.ingest_document("text", "note A", "the same exact content here")
    ingestor.ingest_document("text", "note B", "the same exact content here")
    c = Consolidator(store, cfg)
    assert c.job_dedup() == 1
    live = store.db.execute(
        "SELECT count(*) c FROM memories WHERE archived=0").fetchone()["c"]
    assert live == 1


def test_contradiction_bitemporal(store, cfg, ingestor):
    mid = ingestor.ingest_document("note", "db choice", "we picked a database")
    store.add_relation(mid, "team", "decided", "use MongoDB", valid_at=1000)
    store.add_relation(mid, "team", "decided", "use PostgreSQL", valid_at=2000)
    store.commit()
    c = Consolidator(store, cfg)
    assert c.job_contradictions() == 1
    old = store.db.execute(
        "SELECT * FROM relations WHERE object='use MongoDB'").fetchone()
    assert old["invalid_at"] == 2000          # closed, not deleted
    hist = json.loads(store.get_memory(mid)["history"])
    assert hist and hist[0]["old_value"] == "use MongoDB"


def test_importance_scoring(seeded):
    store, cfg, ing, _ = seeded
    Consolidator(store, cfg).job_importance()
    imps = [r["importance"] for r in
            store.db.execute("SELECT importance FROM memories WHERE archived=0")]
    assert all(0.0 <= i <= 1.0 for i in imps)
    assert any(i > 0.3 for i in imps)  # fresh + decision-bearing memories score


def test_dualmic_reconciliation(seeded):
    store, cfg, ing, ids = seeded
    c = Consolidator(store, cfg, ing)
    assert c.job_dualmic() == 1
    live = store.db.execute(
        """SELECT * FROM memories WHERE source_type='meeting' AND archived=0"""
    ).fetchall()
    assert len(live) == 1
    merged = live[0]
    assert "reconciled" in merged["title"]
    eps = store.db.execute(
        "SELECT * FROM episodes WHERE memory_id=? ORDER BY t_start",
        (merged["memory_id"],)).fetchall()
    # 6 arduino + 1 phone-only tail; the 2 overlapping phone dupes lose on confidence
    assert len(eps) == 7
    assert all(e["asr_confidence"] == pytest.approx(0.92) for e in eps[:-1])
    assert eps[-1]["text"].startswith("Last thing")
    assert eps[-1]["asr_confidence"] == pytest.approx(0.71)


# -------------------------------------------------------------- ask (offline)

def test_ask_offline_fallback(seeded, monkeypatch):
    store, cfg, ing, _ = seeded
    cfg.ollama_url = "http://localhost:1"   # unreachable -> fallback path
    r = Retriever(store, cfg, ing.embedder)
    out = r.ask("Who decided to use JWT authentication?")
    assert out["contexts"]
    assert "LLM unavailable" in out["answer"] or "NOTFOUND" not in out["answer"]
