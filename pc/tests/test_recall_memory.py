"""End-to-end tests for the Recall memory engine (deterministic HashEmbedder,
no Ollama needed)."""
import json

import numpy as np
import pytest

from recall_memory.chunker import _extractive_title, chunk_document, chunk_meeting
from recall_memory.config import CHUNK_SPECS, DIM_COARSE, DIM_FULL, RecallConfig
from recall_memory.consolidate import Consolidator
from recall_memory.extractors import llm_extract_relation
from recall_memory.llm_validate import is_valid_llm_summary, is_valid_relation
from recall_memory.okf import OKFGenerator
from recall_memory.demo_data import (SAMPLE_GITHUB, SAMPLE_OCR, SAMPLE_PDF,
                                     sample_meeting, sample_meeting_phone, seed)
from recall_memory.embeddings import HashEmbedder, matryoshka_coarse
from recall_memory.ingest import Ingestor
from recall_memory.nmo import NMO, Episode, new_id
from recall_memory.retrieval import Retriever
from recall_memory.router import Router
from recall_memory.store import MemoryStore
from recall_memory.tokenizer import count_tokens, tokenize


class FakeLLM:
    """Deterministic stand-in for backend.llm — feeds scripted responses in
    order and records every call so tests can assert on the prompt/schema
    actually used, without needing Ollama."""
    name = "fake"
    model = "fake-model"
    available = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, *, json=False, schema=None, timeout=180.0):
        self.calls.append({"prompt": prompt, "json": json, "schema": schema})
        return self.responses.pop(0)


@pytest.fixture()
def cfg():
    return RecallConfig(db_path=":memory:", backend="hash")


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def ingestor(store, cfg):
    return Ingestor(store, cfg)


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


def test_opening_a_pre_migration_db_backfills_missing_columns(tmp_path):
    """Regression: `CREATE TABLE IF NOT EXISTS` is a no-op on a database file
    that predates a new column (e.g. an existing demo_hub.db from before
    chunks.title existed) — every row is then simply missing that column and
    reading row["title"] raises IndexError, crashing live capture. Simulate
    exactly that: a chunks table built from the schema with `title` (and its
    index) stripped out, then confirm MemoryStore opens it without error and
    the column is present afterward."""
    import sqlite3
    db_path = str(tmp_path / "pre_migration.db")
    raw = sqlite3.connect(db_path)
    raw.execute("""CREATE TABLE chunks (
      chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL,
      chunk_index INTEGER NOT NULL, token_count INTEGER NOT NULL,
      text TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
      episode_ids TEXT DEFAULT '[]', t_start REAL, t_end REAL,
      speaker_span TEXT DEFAULT '[]', source_type TEXT NOT NULL,
      created_at INTEGER NOT NULL, language TEXT DEFAULT 'en',
      importance REAL DEFAULT 0.0, confidence REAL DEFAULT 1.0,
      user TEXT DEFAULT 'default', device TEXT DEFAULT '',
      processing_version TEXT NOT NULL, meeting_id TEXT, repo TEXT,
      file_path TEXT, page INTEGER, document_title TEXT, ocr_confidence REAL,
      emb_full BLOB)""")
    raw.execute("INSERT INTO chunks(memory_id, chunk_index, token_count, text, "
               "char_start, char_end, source_type, created_at, "
               "processing_version) VALUES ('m1',0,1,'x',0,1,'text',1,'v2.0')")
    raw.commit()
    raw.close()

    store = MemoryStore(db_path)   # must not raise
    row = store.db.execute("SELECT * FROM chunks LIMIT 1").fetchone()
    assert "title" in row.keys()
    assert row["title"] == ""
    store.close()


def test_chunk_title_is_extractive_not_parent_title():
    """Every chunk of a memory used to show the SAME parent-memory title in
    citations — now each chunk carries its own short, deterministic label,
    with the speaker prefix stripped for meeting transcript lines."""
    assert _extractive_title("Priya: We decided to use JWT.") \
        == "We decided to use JWT."
    long_title = _extractive_title(
        "Priya: We decided to use JWT authentication with refresh tokens "
        "for the backend API.")
    assert len(long_title) <= 70
    assert long_title.endswith("…")
    assert not long_title.startswith("Priya")


def test_chunk_titles_differ_across_a_long_meeting(seeded):
    store, *_ = seeded
    rows = store.db.execute(
        "SELECT title FROM chunks WHERE source_type='meeting'").fetchall()
    titles = [r["title"] for r in rows]
    assert all(t for t in titles)          # every chunk got a non-empty title
    assert len(set(titles)) > 1            # not all identical


def test_retrieval_exposes_chunk_title(seeded):
    store, cfg, ing, _ = seeded
    r = Retriever(store, cfg, ing.backend)
    ctxs = r.retrieve("Who decided to use JWT authentication?")
    assert ctxs and all(c.chunk_title for c in ctxs)


# -------------------------------------------------------- ollama backend

def test_ollama_llm_disables_thinking_mode(monkeypatch):
    """Regression: gemma4:latest (a "thinking"-capable Ollama model) spent its
    entire num_predict budget on hidden reasoning tokens and returned an
    EMPTY content string (done_reason="length") for a trivial tier-3 call —
    confirmed directly against a running Ollama instance. `think: False` must
    be sent on every call so bounded-latency generation isn't silently eaten
    by hidden reasoning tokens."""
    from recall_memory.backends.ollama import OllamaLLM
    from recall_memory.config import RecallConfig
    import requests
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    def fake_post(url, json=None, timeout=None):
        captured.update(body=json)
        return FakeResp()
    monkeypatch.setattr(requests, "post", fake_post)
    llm = OllamaLLM(RecallConfig(backend="ollama"))
    llm.generate("hello")
    assert captured["body"]["think"] is False


def test_ollama_reranker_is_passthrough_when_unset():
    """Ollama has no rerank endpoint of its own — with no RECALL_RERANKER_MODEL
    configured the fused order must pass straight through, unreordered."""
    from recall_memory.backends.ollama import OllamaBackend
    cfg = RecallConfig(backend="ollama", reranker_model="")
    assert OllamaBackend(cfg).reranker.name == "passthrough"


def test_ollama_reranker_loads_cross_encoder_when_model_configured():
    """RECALL_RERANKER_MODEL set + sentence-transformers installed must select
    the real local cross-encoder, not passthrough — this was silently never
    happening before because the old code read os.environ directly, which
    .env-loaded pydantic-settings values never populate."""
    from recall_memory.backends.ollama import LocalTransformersReranker, OllamaBackend
    cfg = RecallConfig(backend="ollama", reranker_model="BAAI/bge-reranker-v2-m3")
    reranker = OllamaBackend(cfg).reranker
    assert isinstance(reranker, LocalTransformersReranker)
    assert reranker.name == "bge-reranker-local"
    assert reranker.model_name == "BAAI/bge-reranker-v2-m3"


def test_ollama_reranker_falls_back_when_sentence_transformers_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from recall_memory.backends.ollama import OllamaBackend
    cfg = RecallConfig(backend="ollama", reranker_model="BAAI/bge-reranker-v2-m3")
    assert OllamaBackend(cfg).reranker.name == "passthrough"


# ----------------------------------------------------------- embeddings

def test_matryoshka_coarse_shape_and_range():
    emb = HashEmbedder().embed_documents(["hello world", "jwt auth tokens"])
    assert emb.shape == (2, DIM_FULL)
    coarse = matryoshka_coarse(emb)
    assert coarse.shape == (2, DIM_COARSE)
    assert coarse.dtype == np.int8
    assert np.abs(coarse).max() <= 127


# -------------------------------------------------------------- ingestion

def test_extractive_title_kept_when_backend_has_no_llm(store, cfg):
    """HashBackend's NullLLM.available is False, so ingest must never try to
    spawn the async title-refinement thread — the extractive title stands."""
    ingestor = Ingestor(store, cfg)
    mem_id = ingestor.ingest_document(
        "note", "Untitled",
        "Some rambling opening filler before the point. The actual point is "
        "that JWT tokens expire after 30 minutes in this system.")
    row = store.db.execute(
        "SELECT title FROM chunks WHERE memory_id=?", (mem_id,)).fetchone()
    assert row["title"] == _extractive_title(
        "Some rambling opening filler before the point. The actual point is "
        "that JWT tokens expire after 30 minutes in this system.")


def test_ingest_refines_chunk_title_via_background_llm(monkeypatch, store, cfg):
    """When the backend DOES have an available LLM, ingest must fire the
    background refinement pass (run synchronously here by faking out Thread)
    and persist the LLM's title over the extractive placeholder."""
    from recall_memory import ingest as ingest_mod

    class SyncThread:
        def __init__(self, target, args=(), daemon=None):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(ingest_mod.threading, "Thread", SyncThread)

    ingestor = Ingestor(store, cfg)
    fake_llm = FakeLLM(["JWT token expiry policy"])
    ingestor.backend.llm = fake_llm
    mem_id = ingestor.ingest_document(
        "note", "Untitled",
        "Some rambling opening filler before the point. The actual point is "
        "that JWT tokens expire after 30 minutes in this system.")
    row = store.db.execute(
        "SELECT title FROM chunks WHERE memory_id=?", (mem_id,)).fetchone()
    assert row["title"] == "JWT token expiry policy"
    assert fake_llm.calls, "the LLM must actually have been asked for a title"


def test_ingest_keeps_extractive_title_when_llm_response_is_low_quality(
        monkeypatch, store, cfg):
    """A refusal/off-topic LLM answer must not overwrite the extractive title
    — same output-validation discipline as summaries/relations (see
    llm_validate.py): a bad write here can't be corrected later since the
    chunk text isn't re-consulted after the fact."""
    from recall_memory import ingest as ingest_mod

    class SyncThread:
        def __init__(self, target, args=(), daemon=None):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(ingest_mod.threading, "Thread", SyncThread)

    ingestor = Ingestor(store, cfg)
    ingestor.backend.llm = FakeLLM(["I'm a large language model and cannot help with that."])
    text = ("Some rambling opening filler before the point. The actual point "
            "is that JWT tokens expire after 30 minutes in this system.")
    mem_id = ingestor.ingest_document("note", "Untitled", text)
    row = store.db.execute(
        "SELECT title FROM chunks WHERE memory_id=?", (mem_id,)).fetchone()
    assert row["title"] == _extractive_title(text)


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
#
# No more tiers (plan §8's 4-tier router was removed by request — it cost
# real, measured latency — up to 16s/query through a broken tier-3 — for
# classification value the trace logs showed wasn't paying for itself). Every
# query now gets one fixed plan; these tests cover what's left: entity
# extraction (still needed so entity_index/graph aren't always empty) and
# that retrieval runs every route uniformly regardless of query shape.

def test_router_returns_flat_plan_for_every_query(store, cfg):
    """No command detection, no query_type classification, no per-query
    weight profile — every query gets the same fixed shape back."""
    plan = Router(store, cfg).route("forget the last 5 minutes")
    assert plan.query_type == "general"
    assert plan.weight_profile == "general"
    assert plan.filters == {}
    assert plan.needs == {"bm25": True, "vector": True, "graph": True,
                          "metadata_filter": False, "entity_index": True}
    assert plan.rerank is True


def test_router_extracts_known_entities(seeded):
    """Entity extraction survives the tier removal — it's plain term-lookup
    against the entity index, not classification, and both entity_index and
    graph retrieval routes are no-ops without it."""
    store, cfg, ing, _ = seeded
    plan = Router(store, cfg).route("Who decided to use JWT authentication?")
    assert "JWT" in plan.entities


def test_router_same_plan_regardless_of_query_shape(seeded):
    """A decision question, a code question, and a bare greeting all get the
    identical fixed plan now — no more tier-dependent behavior swings."""
    store, cfg, ing, _ = seeded
    r = Router(store, cfg)
    a = r.route("Who decided to use JWT?")
    b = r.route("hi")
    assert a.needs == b.needs
    assert a.weight_profile == b.weight_profile == "general"


# ------------------------------------------------------------- retrieval

def test_retrieval_finds_decision(seeded):
    store, cfg, ing, ids = seeded
    r = Retriever(store, cfg, ing.backend)
    ctxs = r.retrieve("Who decided to use JWT authentication?")
    assert ctxs
    joined = " ".join(c.text.lower() for c in ctxs)
    assert "jwt" in joined
    top_meeting = [c for c in ctxs if c.source_type == "meeting"]
    assert top_meeting, "decision query should surface the meeting"
    assert top_meeting[0].episodes, "meeting context must carry verbatim episodes"


def test_cross_source_bridge(seeded):
    store, cfg, ing, _ = seeded
    r = Retriever(store, cfg, ing.backend)
    ctxs = r.retrieve("JWT", top_k=10)
    types = {c.source_type for c in ctxs}
    assert {"meeting", "github_repo", "pdf"} <= types, f"got {types}"


def test_speaker_filtered_retrieval(seeded):
    store, cfg, ing, _ = seeded
    r = Retriever(store, cfg, ing.backend)
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


# ------------------------------------------------------------------ OKF

def test_okf_generation(seeded):
    store, cfg, ing, _ = seeded
    counts = OKFGenerator(store, cfg).generate_all()
    assert counts["meeting"] >= 1 and counts["github_repo"] >= 1
    mtg = store.get_okf("meeting", "mtg-auth-sync")
    assert mtg is not None
    manifest = json.loads(mtg["manifest"])
    assert "jwt" in " ".join(manifest["decisions"]).lower()
    repo = store.get_okf("github_repo", "recall-backend")
    assert repo is not None
    files = json.loads(repo["manifest"])["files"]
    assert any("jwt_middleware.py" in f for f in files)


def test_okf_regeneration_is_incremental(seeded):
    store, cfg, ing, _ = seeded
    gen = OKFGenerator(store, cfg)
    gen.generate_all()
    # nothing changed -> second pass regenerates nothing
    assert gen.generate_all() == {"github_repo": 0, "meeting": 0, "pdf": 0}


def test_okf_skim_prepended_when_source_dominates(store, cfg, ingestor):
    ingestor.ingest_meeting(sample_meeting())          # meeting-only store
    OKFGenerator(store, cfg).generate_all()
    r = Retriever(store, cfg, ingestor.backend)
    out = r.ask("who decided to use JWT authentication?")
    assert out["okf"] is not None
    assert out["okf"]["meeting_id"] == "mtg-auth-sync"


# ------------------------------------------------ full Dream tier (jobs 2/5/6/8/9)

def test_entity_resolution_merges_variants(store, cfg, ingestor):
    mid = ingestor.ingest_document("text", "n", "placeholder about databases")
    cid = store.db.execute(
        "SELECT chunk_id FROM chunks WHERE memory_id=? LIMIT 1", (mid,)).fetchone()["chunk_id"]
    store.add_entity_mention("PostgreSQL", "identifier", cid, mid)
    store.add_entity_mention("Postgres", "proper_noun", cid, mid)
    store.commit()
    assert Consolidator(store, cfg, ingestor).job_entity_resolution() >= 1
    canon = store.db.execute(
        "SELECT DISTINCT entity_id FROM entity_mentions "
        "WHERE entity_text IN ('PostgreSQL','Postgres') OR entity_id IN "
        "(SELECT entity_id FROM entity_mentions WHERE memory_id=?)", (mid,)).fetchall()
    ids = {r["entity_id"] for r in store.db.execute(
        "SELECT entity_id FROM entity_mentions WHERE memory_id=?", (mid,))}
    assert len(ids) == 1               # both variants collapsed to one id


def test_communities_cluster_shared_entities(seeded):
    store, cfg, ing, _ = seeded
    n = Consolidator(store, cfg, ing).job_communities()
    assert n >= 1                       # JWT bridges the cross-source memories
    rows = store.db.execute("SELECT * FROM communities").fetchall()
    assert rows and json.loads(rows[0]["member_ids"])


def test_summary_ladder(seeded):
    store, cfg, ing, _ = seeded
    assert Consolidator(store, cfg, ing).job_summaries() >= 1
    s = store.get_summary("meeting", "mtg-auth-sync")
    assert s is not None and "jwt" in s["text"].lower()


def test_run_full_reports_every_job(seeded):
    store, cfg, ing, _ = seeded
    out = Consolidator(store, cfg, ing).run_full()
    for key in ("dualmic_merged", "duplicates_merged", "contradictions_closed",
                "importance_updated", "entities_resolved", "communities",
                "summaries", "archived", "graph_repaired", "llm_relations", "okf"):
        assert key in out
    assert out["graph_repaired"] == 0   # nothing orphaned in a clean store


# ------------------------------------------------------- LLM output validation

def test_is_valid_llm_summary_rejects_refusal_and_off_topic():
    facts = "decisions=['use JWT for auth']; participants=['Priya','Rahul']"
    assert not is_valid_llm_summary(
        "I'm a large language model, I don't have personal memories of this "
        "conversation. Please provide the specific text.", facts)
    assert not is_valid_llm_summary(
        "Memory Retrieval Overview\n- a system for retrieving memories", facts)
    assert not is_valid_llm_summary("ok", facts)   # too short
    assert is_valid_llm_summary(
        "The team decided to use JWT for authentication.", facts)


def test_is_valid_relation_rejects_hallucinated_object():
    source = "we decided to use JWT authentication for the backend API"
    assert is_valid_relation("team", "JWT authentication", source)
    assert not is_valid_relation("team", "a year old memory about Ole Miss",
                                source)
    assert not is_valid_relation("", "something", source)


def test_summarize_falls_back_to_extractive_on_invalid_llm_output(store, cfg,
                                                                  ingestor):
    """Reproduces the exact failure: the small model answers with a
    disclaimer instead of a summary; that must NOT get written verbatim —
    the extractive fallback text is kept instead."""
    c = Consolidator(store, cfg, ingestor)
    c.llm = FakeLLM(["I'm a large language model and don't have access to "
                     "that information."])
    extractive = "Decisions: use JWT for auth"
    assert c._summarize(extractive, extractive) == extractive


def test_okf_llm_polish_falls_back_on_disclaimer(store, cfg):
    gen = OKFGenerator(store, cfg, llm=FakeLLM(["As an AI language model, I "
                                                "cannot access your memories."]))
    facts = "meeting_id=mtg-1; decisions=['use JWT']"
    assert gen._maybe_summarize("fallback text", facts) == "fallback text"


def test_llm_extract_relation_rejects_hallucinated_triple():
    # object shares NO vocabulary with the sentence it was supposedly
    # extracted from -> the heuristic's actual catchable case (a hallucinated
    # triple whose garbage happens to echo the source's own garbage, e.g. two
    # different "Ole Miss" mentions from the same ASR hallucination loop, is
    # a known limitation — see llm_validate.py's docstring).
    fake = FakeLLM(['{"subject": "memory retrieval", "predicate": "works", '
                    '"object": "a completely unrelated topic never mentioned"}'])
    sentence = "we decided to use JWT authentication for the backend API"
    assert llm_extract_relation(sentence, fake) is None


def test_llm_extract_relation_accepts_on_topic_triple():
    fake = FakeLLM(['{"subject": "team", "predicate": "decided", '
                    '"object": "use JWT authentication"}'])
    sentence = "we decided to use JWT authentication for the backend API"
    rel = llm_extract_relation(sentence, fake)
    assert rel == {"subject": "team", "predicate": "decided",
                   "object": "use JWT authentication"}


# ---------------------------------------------------- job 7: re-embedding

def test_job_reembed_updates_stale_chunks(seeded):
    store, cfg, ing, _ = seeded
    store.db.execute("UPDATE chunks SET processing_version='v0.1-old'")
    store.commit()
    n_stale = store.stats()["chunks"]
    n = Consolidator(store, cfg, ing).job_reembed()
    assert n == n_stale
    from recall_memory.config import PROCESSING_VERSION
    versions = {r["processing_version"] for r in store.db.execute(
        "SELECT DISTINCT processing_version FROM chunks")}
    assert versions == {PROCESSING_VERSION}
    # idempotent — nothing left to re-embed on a second pass
    assert Consolidator(store, cfg, ing).job_reembed() == 0


# -------------------------------------------------------------- ask (offline)

def test_ask_offline_fallback(seeded, monkeypatch):
    store, cfg, ing, _ = seeded
    cfg.ollama_url = "http://localhost:1"   # unreachable -> fallback path
    r = Retriever(store, cfg, ing.backend)
    out = r.ask("Who decided to use JWT authentication?")
    assert out["contexts"]
    assert "LLM unavailable" in out["answer"] or "NOTFOUND" not in out["answer"]
