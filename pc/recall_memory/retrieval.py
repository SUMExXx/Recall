"""Retrieval pipeline — plan §9-§11.

needs{}-gated parallel indices -> dedup -> weighted RRF (k=60) ->
recency/importance boost -> float[768] rescore of the fused top-200 ->
reranker hook (Qwen3-Reranker slot; passthrough here) -> small-to-big
expansion -> optional LLM answer with citations.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from .backends import get_backend
from .config import (ANSWER_TOPK, COARSE_TOPK, DIM_FULL, RECENCY_HALF_LIFE_DAYS,
                     RERANK_TOPK, RRF_K, WEIGHT_PROFILES, RecallConfig)
from .embeddings import matryoshka_coarse
from .router import RoutePlan, Router
from .store import MemoryStore
from .tracing import ensure_trace, step

log = logging.getLogger("recall.retrieval")

# How much the LLM listwise reranker's own score counts vs. the float cosine
# when the two are blended (the rest is cosine). The small on-device LLM's
# absolute relevance scores are noisy, so cosine is kept as an equal anchor.
RERANK_BLEND = 0.5

_CITATION_RE = re.compile(r"\s*\[[0-9a-f]{6,10}:\d+\]", re.IGNORECASE)


class _SpokenTextFilter:
    """Strips inline citations like "[8f52a72:1]" out of an LLM's streamed
    deltas before they reach TTS — read aloud verbatim they come out as
    garbled digits ("eight eff five two a seven two colon one"), which is
    exactly the "numbers and noise" a listener notices. A citation arrives
    one character at a time, so a per-delta regex can't tell yet whether a
    "[" starts one until it closes; this buffers from the last unclosed "["
    onward until it resolves, mirroring the dashboard's own
    stripPartialCitation() for the visible transcript."""

    def __init__(self):
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        last_open, last_close = self._buf.rfind("["), self._buf.rfind("]")
        if last_open > last_close:
            emit = _CITATION_RE.sub("", self._buf[:last_open])
            self._buf = self._buf[last_open:]
            return emit
        emit = _CITATION_RE.sub("", self._buf)
        self._buf = ""
        return emit

    def flush(self) -> str:
        emit = _CITATION_RE.sub("", self._buf)
        self._buf = ""
        return emit


@dataclass
class RetrievedContext:
    chunk_id: int
    memory_id: str
    score: float
    text: str
    source_type: str
    title: str              # parent MEMORY's title — for grouping/citation source
    summary: str
    chunk_title: str = ""   # this CHUNK's own extractive title (chunker.py) —
                            # what a given hit is actually about, distinct from
                            # every other chunk of the same long memory
    neighbors: list[str] = field(default_factory=list)
    episodes: list[dict] = field(default_factory=list)   # verbatim citations
    meta: dict = field(default_factory=dict)

    def citation(self) -> str:
        return f"[{self.memory_id[:8]}:{self.chunk_id}]"


def _fmt_ranked(items, n: int = 6) -> str:
    """Render up to N (chunk_id, score, [text]) results as a compact, ordered
    string for the trace — e.g. "12:0.812, 7:0.640(...), 3:0.511" — so a trace
    read can compare exactly which candidates/scores survived at EACH stage,
    not just how many. Every call site passes data already computed for that
    stage (no extra DB reads) so this is pure formatting, not new I/O."""
    if not items:
        return "(none)"
    out = []
    for it in items[:n]:
        cid, sc, *rest = it
        text = f" {rest[0][:36]!r}" if rest and rest[0] else ""
        out.append(f"{cid}:{sc:.3f}{text}")
    more = f"  (+{len(items) - n} more)" if len(items) > n else ""
    return ", ".join(out) + more


def _rrf_fuse(route_lists: dict[str, list[tuple[int, float]]],
              weights: dict[str, float]) -> dict[int, float]:
    """Weighted RRF: score(d) = sum over routes of w_r / (k + rank_r(d))."""
    fused: dict[int, float] = {}
    for route, results in route_lists.items():
        w = weights.get(route, 0.5)
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + w / (RRF_K + rank)
    return fused


class Retriever:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 backend=None, reranker=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.backend = backend or get_backend(self.cfg)
        self.embedder = self.backend.embedder
        self.reranker = reranker or self.backend.reranker
        self.llm = self.backend.llm
        self.router = Router(store, self.cfg)

    # ------------------------------------------------------------ pipeline

    def retrieve(self, query: str, plan: RoutePlan | None = None,
                 top_k: int = ANSWER_TOPK) -> list[RetrievedContext]:
        """Public entry point — opens its own trace when called standalone
        (`/search`, CLI `search`, MCP `recall_search_memory`); nests inside the
        caller's trace when called from `ask()`."""
        with ensure_trace("retrieve", query=query, top_k=top_k):
            return self._retrieve_impl(query, plan, top_k)

    def _retrieve_impl(self, query: str, plan: RoutePlan | None, top_k: int,
                       query_vec: np.ndarray | None = None
                       ) -> list[RetrievedContext]:
        if query_vec is None:   # ask() passes its own vec — one embed per request
            with step("embed_query", text=query) as s:
                query_vec = self.embedder.embed_query(query)
                s.detail(embedder=self.embedder.name, dim=int(query_vec.shape[0]))
        if plan is None:
            plan = self.router.route(query)

        needs, filters = plan.needs, plan.filters
        t0 = time.perf_counter()
        route_lists: dict[str, list[tuple[int, float]]] = {}
        if needs.get("bm25"):
            with step("retrieve:bm25", filters=filters) as s:
                route_lists["bm25"] = self.store.bm25_search(query, filters)
                s.detail(hits=len(route_lists["bm25"]), top=_fmt_ranked(route_lists["bm25"]))
        if needs.get("vector"):
            with step("retrieve:vector", k=COARSE_TOPK) as s:
                coarse_q = matryoshka_coarse(query_vec[np.newaxis, :])[0]
                route_lists["vector"] = self.store.vector_search(
                    coarse_q, filters, k=COARSE_TOPK)
                s.detail(hits=len(route_lists["vector"]), top=_fmt_ranked(route_lists["vector"]))
        if needs.get("entity_index"):
            with step("retrieve:entity_index", entities=plan.entities) as s:
                route_lists["entity"] = self.store.entity_search(plan.entities, filters)
                s.detail(hits=len(route_lists["entity"]), top=_fmt_ranked(route_lists["entity"]))
        if needs.get("metadata_filter") and filters:
            with step("retrieve:metadata_filter", filters=filters) as s:
                route_lists["metadata"] = self.store.metadata_search(filters)
                s.detail(hits=len(route_lists["metadata"]), top=_fmt_ranked(route_lists["metadata"]))
        if needs.get("graph"):
            with step("retrieve:graph", entities=plan.entities) as s:
                route_lists["graph"] = self.store.graph_search(plan.entities, filters)
                s.detail(hits=len(route_lists["graph"]), top=_fmt_ranked(route_lists["graph"]))
        log.debug("routes %s in %.0f ms",
                  {k: len(v) for k, v in route_lists.items()},
                  (time.perf_counter() - t0) * 1000)

        profile = WEIGHT_PROFILES.get(plan.weight_profile, WEIGHT_PROFILES["general"])
        with step("fuse_rrf", weight_profile=plan.weight_profile) as s:
            fused = _rrf_fuse(route_lists, profile["routes"])
            top_fused = sorted(fused.items(), key=lambda kv: -kv[1])
            s.detail(candidates=len(fused), top=_fmt_ranked(top_fused))
        if not fused:
            return []

        with step("boost_recency_importance") as s:
            boosted = self._boost(fused, profile["boost"])
            top_boosted = sorted(boosted.items(), key=lambda kv: -kv[1])
            s.detail(boost_weights=profile["boost"], top=_fmt_ranked(top_boosted))
        top = sorted(boosted.items(), key=lambda kv: -kv[1])[:COARSE_TOPK]
        with step(f"rescore_float{DIM_FULL}_cosine") as s:
            scored = self._rescore(top, query_vec)
            # (chunk_id, blended_score, cosine) -> log both, cosine is what the
            # relevance_cutoff below actually gates vector-only candidates on.
            s.detail(candidates=len(scored),
                     top=_fmt_ranked([(cid, sc, f"cos={cos:.3f}") for cid, sc, cos in scored]))
        cos_of = {cid: cos for cid, _s, cos in scored}
        rescored = [(cid, s) for cid, s, _cos in scored]

        rerank_scores: dict[int, float] = {}
        if plan.rerank:
            with step("rerank", reranker=self.reranker.name) as s:
                cands_chunk_ids = [cid for cid, _ in rescored[:RERANK_TOPK]]
                cands_chunks = self.store.get_chunks_batch(cands_chunk_ids)
                cands = []
                for cid, sc in rescored[:RERANK_TOPK]:
                    crow = cands_chunks.get(cid)
                    if crow:
                        cands.append((cid, sc, crow["text"]))
                reranked = self.reranker.rerank(query, cands)
                # The on-device LLM reranker's ABSOLUTE listwise scores are
                # noisy — it has ranked an off-topic passage above the right
                # one (e.g. a "favorite bike" chunk at 0.99 for "what was my
                # project"). Anchor each score to the float cosine already
                # computed at 50/50: a candidate the LLM loves but cosine says
                # is unrelated gets pulled back, and vice-versa. Passthrough
                # returns the fused scores unchanged, so only a real (LLM /
                # cross-encoder) reranker is blended.
                if self.reranker.name != "passthrough":
                    reranked = sorted(
                        ((cid, RERANK_BLEND * rs
                          + (1 - RERANK_BLEND) * cos_of.get(cid, 0.0))
                         for cid, rs in reranked),
                        key=lambda t: -t[1])
                rerank_scores = dict(reranked)
                rescored = reranked + rescored[RERANK_TOPK:]
                text_of = {cid: text for cid, _sc, text in cands}
                s.detail(candidates=len(cands),
                         top=_fmt_ranked([(cid, sc, text_of.get(cid, "")) for cid, sc in reranked]))

        # Relevance cutoff. Two independent signals, either can drop a
        # candidate:
        #  - cross-encoder: a REAL reranker (not passthrough) already judged
        #    this (query, chunk) pair — a candidate it scored near-zero must
        #    not reach the LLM just because it also picked up an incidental
        #    BM25/entity hit (e.g. one shared stopword). Without this, the
        #    reranker's score was computed but never actually gated anything.
        #  - vector-only: a candidate whose ONLY support is the vector route
        #    needs a cosine competitive with the best hit — RRF rank share
        #    alone lets unrelated chunks ride in at ~0.5 when the pool is small.
        with step("relevance_cutoff") as s:
            is_real_reranker = bool(rerank_scores) and self.reranker.name != "passthrough"
            top_rerank = max(rerank_scores.values(), default=0.0)
            evidence = {cid for route, results in route_lists.items()
                        if route != "vector" for cid, _ in results}
            top_cos = max(cos_of.values(), default=0.0)
            before = len(rescored)
            kept, dropped_ids = [], []
            for cid, sc in rescored:
                if is_real_reranker and cid in rerank_scores:
                    rs = rerank_scores[cid]
                    if rs < 0.1 or rs < 0.25 * top_rerank:
                        dropped_ids.append(cid)
                        continue
                elif top_cos > 0:
                    is_vec_only = cid not in evidence
                    cos = cos_of.get(cid, 0.0)
                    if is_vec_only and not (cos >= 0.40 and cos >= 0.6 * top_cos):
                        dropped_ids.append(cid)
                        continue
                kept.append((cid, sc))
            if kept:
                rescored = kept
            s.detail(before=before, after=len(rescored), dropped=before - len(rescored),
                     dropped_ids=dropped_ids[:10], top=_fmt_ranked(rescored))

        with step("small_to_big_expand", top_k=top_k) as s:
            out = [self._expand(cid, sc) for cid, sc in rescored[:top_k]]
            s.detail(top=_fmt_ranked([(c.chunk_id, c.score, c.chunk_title or c.title)
                                      for c in out]))
            s.detail(contexts=len(out))
        return out

    def _boost(self, fused: dict[int, float], boost_w: dict) -> dict[int, float]:
        """Post-fusion multiplicative recency/importance boost (plan §9 step 3)."""
        now = time.time()
        half_life = RECENCY_HALF_LIFE_DAYS * 86400.0
        out: dict[int, float] = {}
        chunks_map = self.store.get_chunks_batch(list(fused.keys()))
        for cid, score in fused.items():
            row = chunks_map.get(cid)
            if row is None:
                continue
            age = max(0.0, now - row["created_at"])
            recency = math.pow(0.5, age / half_life)
            factor = (1.0 + boost_w.get("recency", 0.0) * recency
                      + boost_w.get("importance", 0.0) * (row["importance"] or 0.0))
            out[cid] = score * factor
        return out

    def _rescore(self, top: list[tuple[int, float]],
                 query_vec: np.ndarray) -> list[tuple[int, float, float]]:
        """Exact float[768] cosine over the fused top-K; blended with RRF.
        Returns (chunk_id, blended_score, cosine) — the raw cosine feeds the
        relevance cutoff in retrieve()."""
        if not top:
            return []
        max_rrf = max(s for _, s in top) or 1.0
        out = []
        chunk_ids = [cid for cid, _ in top]
        chunks_map = self.store.get_chunks_batch(chunk_ids)
        for cid, rrf_score in top:
            row = chunks_map.get(cid)
            if row is None:
                continue
            cos = 0.0
            if row["emb_full"]:
                emb = np.frombuffer(row["emb_full"], dtype=np.float32)
                if emb.shape[0] == DIM_FULL:
                    cos = float(np.dot(emb, query_vec))
            cos = max(0.0, cos)
            out.append((cid, 0.5 * (rrf_score / max_rrf) + 0.5 * cos, cos))
        out.sort(key=lambda t: -t[1])
        return out

    def _expand(self, chunk_id: int, score: float) -> RetrievedContext:
        """Small-to-big: chunk -> neighbor windows -> parent NMO -> episodes."""
        row = self.store.get_chunk(chunk_id)
        mem = self.store.get_memory(row["memory_id"])
        episodes = [
            {"episode_id": e["episode_id"], "speaker": e["speaker"],
             "t_start": e["t_start"], "t_end": e["t_end"], "text": e["text"]}
            for e in self.store.episodes_for_chunk(row)]
        return RetrievedContext(
            chunk_id=chunk_id, memory_id=row["memory_id"], score=score,
            text=row["text"], source_type=row["source_type"],
            title=mem["title"] if mem else "", summary=mem["summary"] if mem else "",
            chunk_title=row["title"] or "",
            neighbors=[n["text"] for n in self.store.neighbor_chunks(row)],
            episodes=episodes,
            meta={k: row[k] for k in ("meeting_id", "repo", "file_path", "page",
                                      "document_title", "speaker_span", "created_at")
                  if row[k] is not None})

    # -------------------------------------------------------------- answer

    def ask(self, query: str, top_k: int = ANSWER_TOPK) -> dict:
        """Route -> retrieve -> synthesize (LLM) with citations.

        Returns {answer, plan, contexts}. Falls back to returning contexts
        verbatim when no LLM is reachable. Every stage is recorded to the
        step trace (logs/recall_trace.log by default) via ensure_trace/step.
        """
        with ensure_trace("ask", query=query, top_k=top_k):
            return self._ask_impl(query, top_k)

    def _ask_impl(self, query: str, top_k: int) -> dict:
        t0 = time.perf_counter()
        with step("embed_query", text=query) as s:
            query_vec = self.embedder.embed_query(query)
            s.detail(embedder=self.embedder.name)
        t_embed = time.perf_counter()
        plan = self.router.route(query)   # instruments its own step
        # same trace, reuse query_vec — no redundant second embed call
        contexts = self._retrieve_impl(query, plan, top_k, query_vec=query_vec)
        t_retrieve = time.perf_counter()
        if not contexts:
            log.info("ask %r: no matches (embed %.0f ms, retrieve %.0f ms)",
                     query[:50], (t_embed - t0) * 1000,
                     (t_retrieve - t_embed) * 1000)
            return {"answer": self._no_context_answer(query),
                    "plan": plan, "contexts": [], "okf": None}
        with step("okf_lookup") as s:
            okf = self._okf_for_contexts(contexts)
            s.detail(attached=okf is not None)
        answer = self._synthesize(query, contexts, okf)
        log.info("ask %r: %s · embed %.0f ms · retrieve %.0f ms · "
                 "synth %.0f ms · %d ctx%s", query[:50],
                 plan.query_type, (t_embed - t0) * 1000,
                 (t_retrieve - t_embed) * 1000,
                 (time.perf_counter() - t_retrieve) * 1000,
                 len(contexts), " · okf" if okf else "")
        return {"answer": answer, "plan": plan, "contexts": contexts, "okf": okf}

    _NO_CONTEXT_PROMPT = (
        "You are Recall, a personal on-device memory assistant. The user "
        "asked a question below, but no relevant memory was found for it. "
        "Reply with one short, natural sentence saying you don't have "
        "anything recorded about this yet. Do not guess, do not invent "
        "facts, do not apologize at length.\n\nQUESTION: {query}\nANSWER:")

    def _no_context_answer(self, query: str) -> str:
        """Still speaks through the LLM when possible — a bare "NOTFOUND"
        string reads as a broken assistant, not an honest "I don't know
        that yet" from one. Never fabricates facts either way: with no
        contexts there is nothing for the LLM to draw on but the question
        itself, and the prompt explicitly forbids guessing."""
        fallback = "I don't have anything recorded about that yet."
        if not self.llm.available:
            return fallback
        try:
            return self.llm.generate(
                self._NO_CONTEXT_PROMPT.format(query=query), timeout=15.0).strip() or fallback
        except Exception:
            return fallback

    def ask_stream(self, query: str, top_k: int = ANSWER_TOPK, tts=None):
        """Like ask(), but yields incremental events instead of one blob —
        for a UI that wants to show the answer typing out and (optionally)
        hear it spoken before the whole thing has finished generating:

          {"type": "delta", "text": "..."}   — next chunk of the LLM answer
          {"type": "audio", "audio": bytes}  — TTS audio chunk, if `tts` given
          {"type": "done", "answer":, "plan":, "contexts":, "okf":}

        `tts`, when given, must expose stream_synthesize(text_chunks_iter)
        -> Iterator[bytes] (see hub/tts.py). The SAME deltas driving the
        visible transcript are fed to it live, on a background thread, so
        the first sentence can be playing while later ones are still being
        generated — one LLM pass drives both the transcript and the voice.
        Falls back to a single delta + done when the backend has no
        streaming LLM (generate_stream raises/unavailable) or no contexts
        were found.
        """
        with ensure_trace("ask", query=query, top_k=top_k, streaming=True):
            yield from self._ask_stream_impl(query, top_k, tts)

    def _ask_stream_impl(self, query: str, top_k: int, tts):
        with step("embed_query", text=query) as s:
            query_vec = self.embedder.embed_query(query)
            s.detail(embedder=self.embedder.name)
        plan = self.router.route(query)
        contexts = self._retrieve_impl(query, plan, top_k, query_vec=query_vec)
        if not contexts:
            answer = self._no_context_answer(query)
            yield {"type": "delta", "text": answer}
            yield {"type": "done", "answer": answer, "plan": plan,
                  "contexts": [], "okf": None}
            return
        with step("okf_lookup") as s:
            okf = self._okf_for_contexts(contexts)
            s.detail(attached=okf is not None)

        if not self.llm.available:
            answer = self._synthesize(query, contexts, okf)
            yield {"type": "delta", "text": answer}
            yield {"type": "done", "answer": answer, "plan": plan,
                  "contexts": contexts, "okf": okf}
            return

        prompt = self._build_prompt(query, contexts, okf)
        parts: list[str] = []
        import queue as _queue
        audio_q: _queue.Queue = _queue.Queue()
        _done = object()
        tts_thread = None

        if tts is not None:
            import threading

            text_q: _queue.Queue = _queue.Queue()
            spoken = _SpokenTextFilter()

            def _text_source():
                while True:
                    item = text_q.get()
                    if item is _done:
                        return
                    yield item

            def _pump():
                try:
                    for chunk in tts.stream_synthesize(_text_source()):
                        audio_q.put(chunk)
                except Exception:
                    log.warning("ask_stream: tts pump failed", exc_info=True)
                finally:
                    audio_q.put(_done)

            tts_thread = threading.Thread(target=_pump, daemon=True)
            tts_thread.start()

        # `_done` off audio_q is consumed exactly once by whichever loop
        # below sees it first — this flag stops the OTHER loop from then
        # blocking on a sentinel that's already gone (a real bug caught
        # while writing this: the final drain used to `get(timeout=30)`
        # forever if the mid-stream opportunistic drain had already
        # swallowed the one-and-only `_done`).
        tts_finished = tts is None

        with step("llm_synthesize_stream", backend=self.backend.name,
                  model=getattr(self.llm, "model", self.llm.name),
                  prompt_chars=len(prompt)) as s:
            try:
                for delta in self.llm.generate_stream(prompt):
                    parts.append(delta)
                    if tts is not None:
                        spoken_text = spoken.feed(delta)
                        if spoken_text:
                            text_q.put(spoken_text)
                    yield {"type": "delta", "text": delta}
                    if tts is not None and not tts_finished:
                        while True:
                            try:
                                item = audio_q.get_nowait()
                            except _queue.Empty:
                                break
                            if item is _done:
                                tts_finished = True
                                break
                            yield {"type": "audio", "audio": item}
            except Exception as e:
                s.detail(fallback=f"LLM stream unavailable: {e}")
                if not parts:   # nothing streamed yet — fall back cleanly
                    answer = self._synthesize(query, contexts, okf)
                    parts = [answer]
                    yield {"type": "delta", "text": answer}
            s.detail(response="".join(parts))

        if tts is not None:
            trailing = spoken.flush()
            if trailing:
                text_q.put(trailing)
            text_q.put(_done)
            if not tts_finished:
                while True:
                    item = audio_q.get(timeout=30.0)
                    if item is _done:
                        break
                    yield {"type": "audio", "audio": item}
            tts_thread.join(timeout=1.0)

        yield {"type": "done", "answer": "".join(parts), "plan": plan,
              "contexts": contexts, "okf": okf}

    def _okf_for_contexts(self, contexts: list[RetrievedContext]) -> dict | None:
        """Skim-the-README (plan §7): if one source root dominates the hits and
        it has an OKF manifest, hand that to the LLM first.

        Dominance by CHUNK COUNT (>=2 contexts, or all of them, sharing a
        source) is the original signal — it fires for a long meeting/PDF where
        several hits land in the same document. It structurally can never fire
        for a corpus of many single-chunk memories (each is its own "source",
        so no root ever repeats) — which is why `okf_lookup attached=False` on
        every ask against exactly that kind of corpus isn't a bug on its own.
        Added: dominance by SCORE — when the #1 hit's score clearly leads the
        rest, the answer is effectively about that one memory even though it
        contributes only one chunk, so that memory's own OKF is still useful
        framing. Purely additive: only creates NEW attachment opportunities."""
        from collections import Counter
        keymap = {"meeting": "meeting_id", "github_repo": "repo",
                  "pdf": "document_title"}
        roots: Counter = Counter()
        for c in contexts:
            key = keymap.get(c.source_type)
            val = c.meta.get(key) if key else None
            if val:
                roots[(c.source_type, val)] += 1
        if not roots:
            return None
        total = len(contexts)
        (stype, skey), n = roots.most_common(1)[0]
        dominant = n == total or n >= max(2, (total + 1) // 2)
        if not dominant and contexts:
            top = contexts[0]
            key = keymap.get(top.source_type)
            val = top.meta.get(key) if key else None
            second = contexts[1].score if len(contexts) > 1 else 0.0
            if val and top.score > 0 and (second <= 0 or top.score >= 1.5 * second):
                stype, skey, dominant = top.source_type, val, True
        if not dominant:
            return None
        row = self.store.get_okf(stype, skey)
        return json.loads(row["manifest"]) if row else None

    def _build_prompt(self, query: str, contexts: list[RetrievedContext],
                      okf: dict | None = None) -> str:
        blocks = []
        for c in contexts:
            # chunk_title (this hit's OWN extractive label) over the parent
            # memory's title — long memories have many chunks about different
            # things; showing the memory title for all of them is what made
            # every citation in a synthesis prompt look the same.
            src = c.chunk_title or c.title or c.source_type
            body = c.text
            if c.episodes:  # verbatim, speaker-attributed quotes for meetings
                body = "\n".join(f'{e["speaker"]}: {e["text"]}' for e in c.episodes)
            # bounded per-context budget — prompt length is synth latency
            blocks.append(f"--- {c.citation()} ({c.source_type}: {src})\n{body[:700]}")
        context_str = "\n".join(blocks)
        okf_block = ""
        if okf:
            okf_block = ("SOURCE OVERVIEW (skim this table of contents first):\n"
                         + json.dumps(okf, ensure_ascii=False)[:1200] + "\n\n")
        return (
            "You are Recall, a personal on-device memory assistant. Answer the "
            "QUESTION using ONLY facts stated in the MEMORY EXCERPTS below — no "
            "outside knowledge, no guessing. Write one short, direct, "
            "natural-sounding answer in your own words — do not just copy an "
            "excerpt verbatim, and ignore any filler or rambling in an excerpt "
            "that isn't actually about the question. Cite the excerpt each "
            "fact came from inline, like [abc12345:3]. If the excerpts don't "
            "actually answer the question, reply exactly NOTFOUND.\n\n"
            f"{okf_block}MEMORY EXCERPTS:\n{context_str}\n\nQUESTION: {query}\nANSWER:")

    def _synthesize(self, query: str, contexts: list[RetrievedContext],
                    okf: dict | None = None) -> str:
        prompt = self._build_prompt(query, contexts, okf)
        with step("llm_synthesize", backend=self.backend.name,
                  model=getattr(self.llm, "model", self.llm.name),
                  prompt_chars=len(prompt)) as s:
            s.detail(prompt=prompt)
            try:
                answer = self.llm.generate(prompt)
                s.detail(response=answer)
                return answer
            except Exception as e:
                s.detail(fallback=f"LLM unavailable: {e}")
                # Offline / no-LLM fallback: hand back the raw evidence.
                return "LLM unavailable — top matches:\n" + "\n".join(
                    f"{c.citation()} {c.text[:160]}" for c in contexts)
