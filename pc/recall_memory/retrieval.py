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
                s.detail(hits=len(route_lists["bm25"]))
        if needs.get("vector"):
            with step("retrieve:vector", k=COARSE_TOPK) as s:
                coarse_q = matryoshka_coarse(query_vec[np.newaxis, :])[0]
                route_lists["vector"] = self.store.vector_search(
                    coarse_q, filters, k=COARSE_TOPK)
                s.detail(hits=len(route_lists["vector"]))
        if needs.get("entity_index"):
            with step("retrieve:entity_index", entities=plan.entities) as s:
                route_lists["entity"] = self.store.entity_search(plan.entities, filters)
                s.detail(hits=len(route_lists["entity"]))
        if needs.get("metadata_filter") and filters:
            with step("retrieve:metadata_filter", filters=filters) as s:
                route_lists["metadata"] = self.store.metadata_search(filters)
                s.detail(hits=len(route_lists["metadata"]))
        if needs.get("graph"):
            with step("retrieve:graph", entities=plan.entities) as s:
                route_lists["graph"] = self.store.graph_search(plan.entities, filters)
                s.detail(hits=len(route_lists["graph"]))
        log.debug("routes %s in %.0f ms",
                  {k: len(v) for k, v in route_lists.items()},
                  (time.perf_counter() - t0) * 1000)

        profile = WEIGHT_PROFILES.get(plan.weight_profile, WEIGHT_PROFILES["general"])
        with step("fuse_rrf", weight_profile=plan.weight_profile) as s:
            fused = _rrf_fuse(route_lists, profile["routes"])
            s.detail(candidates=len(fused))
        if not fused:
            return []

        with step("boost_recency_importance") as s:
            boosted = self._boost(fused, profile["boost"])
            s.detail(boost_weights=profile["boost"])
        top = sorted(boosted.items(), key=lambda kv: -kv[1])[:COARSE_TOPK]
        with step("rescore_float768_cosine") as s:
            scored = self._rescore(top, query_vec)
            s.detail(candidates=len(scored))
        cos_of = {cid: cos for cid, _s, cos in scored}
        rescored = [(cid, s) for cid, s, _cos in scored]

        if plan.rerank:
            with step("rerank", reranker=self.reranker.name) as s:
                cands_chunk_ids = [cid for cid, _ in rescored[:RERANK_TOPK]]
                cands_chunks = self.store.get_chunks_batch(cands_chunk_ids)
                cands = []
                for cid, sc in rescored[:RERANK_TOPK]:
                    crow = cands_chunks.get(cid)
                    if crow:
                        cands.append((cid, sc, crow["text"]))
                rescored = self.reranker.rerank(query, cands) + rescored[RERANK_TOPK:]
                s.detail(candidates=len(cands))

        # Relevance cutoff: a candidate whose ONLY support is the vector route
        # needs a cosine competitive with the best hit — RRF rank share alone
        # lets unrelated chunks ride in at ~0.5 when the pool is small.
        with step("relevance_cutoff") as s:
            evidence = {cid for route, results in route_lists.items()
                        if route != "vector" for cid, _ in results}
            top_cos = max(cos_of.values(), default=0.0)
            before = len(rescored)
            if top_cos > 0:
                kept = []
                for cid, sc in rescored:
                    is_vec_only = cid not in evidence
                    cos = cos_of.get(cid, 0.0)
                    if not is_vec_only or (cos >= 0.40 and cos >= 0.6 * top_cos):
                        kept.append((cid, sc))
                if kept:
                    rescored = kept
            s.detail(before=before, after=len(rescored), dropped=before - len(rescored))

        with step("small_to_big_expand", top_k=top_k) as s:
            out = [self._expand(cid, sc) for cid, sc in rescored[:top_k]]
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
            log.info("ask %r: NOTFOUND (embed %.0f ms, retrieve %.0f ms)",
                     query[:50], (t_embed - t0) * 1000,
                     (t_retrieve - t_embed) * 1000)
            return {"answer": "NOTFOUND — no matching memories.",
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

    def _okf_for_contexts(self, contexts: list[RetrievedContext]) -> dict | None:
        """Skim-the-README (plan §7): if one source root dominates the hits and
        it has an OKF manifest, hand that to the LLM first."""
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
        # attach when the hits are essentially about one source: either every
        # context shares the root, or it holds a clear majority (>=2).
        if not (n == total or n >= max(2, (total + 1) // 2)):
            return None
        row = self.store.get_okf(stype, skey)
        return json.loads(row["manifest"]) if row else None

    def _synthesize(self, query: str, contexts: list[RetrievedContext],
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
        prompt = (
            "You are Recall, an on-device memory assistant. Answer the question "
            "using ONLY the memory excerpts below. Cite the source of each fact you use in your answer "
            "of the excerpts you used. If the excerpts do not contain the "
            "answer, reply exactly NOTFOUND.\n\n"
            f"{okf_block}MEMORY EXCERPTS:\n{context_str}\n\nQUESTION: {query}\nANSWER:")
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
