"""Retrieval pipeline — plan §9-§11.

needs{}-gated parallel indices -> dedup -> weighted RRF (k=60) ->
recency/importance boost -> float[768] rescore of the fused top-200 ->
reranker hook (Qwen3-Reranker slot; passthrough here) -> small-to-big
expansion -> optional LLM answer with citations.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .config import (ANSWER_TOPK, COARSE_TOPK, DIM_FULL, RECENCY_HALF_LIFE_DAYS,
                     RERANK_TOPK, RRF_K, WEIGHT_PROFILES, RecallConfig)
from .embeddings import get_embedder, matryoshka_coarse
from .router import RoutePlan, Router
from .store import MemoryStore


@dataclass
class RetrievedContext:
    chunk_id: int
    memory_id: str
    score: float
    text: str
    source_type: str
    title: str
    summary: str
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


class PassthroughReranker:
    """Slot for Qwen3-Reranker-0.6B on the event PC (bge-reranker CPU fallback)."""

    def rerank(self, query: str, candidates: list[tuple[int, float, str]]
               ) -> list[tuple[int, float]]:
        return [(cid, score) for cid, score, _text in candidates]


class Retriever:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 embedder=None, reranker=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.embedder = embedder or get_embedder(self.cfg)
        self.reranker = reranker or PassthroughReranker()
        self.router = Router(store, self.cfg, self.embedder)

    # ------------------------------------------------------------ pipeline

    def retrieve(self, query: str, plan: RoutePlan | None = None,
                 top_k: int = ANSWER_TOPK) -> list[RetrievedContext]:
        query_vec = self.embedder.embed_query(query)
        if plan is None:
            plan = self.router.route(query, query_vec)
        if plan.command:
            return []  # tier-0 commands are handled by the caller, not retrieval

        needs, filters = plan.needs, plan.filters
        route_lists: dict[str, list[tuple[int, float]]] = {}
        if needs.get("bm25"):
            route_lists["bm25"] = self.store.bm25_search(query, filters)
        if needs.get("vector"):
            coarse_q = matryoshka_coarse(query_vec[np.newaxis, :])[0]
            route_lists["vector"] = self.store.vector_search(
                coarse_q, filters, k=COARSE_TOPK)
        if needs.get("entity_index"):
            route_lists["entity"] = self.store.entity_search(plan.entities, filters)
        if needs.get("metadata_filter") and (filters or plan.path == "fast"):
            route_lists["metadata"] = self.store.metadata_search(filters)
        if needs.get("graph"):
            route_lists["graph"] = self.store.graph_search(plan.entities, filters)

        profile = WEIGHT_PROFILES.get(plan.weight_profile, WEIGHT_PROFILES["general"])
        fused = _rrf_fuse(route_lists, profile["routes"])
        if not fused:
            return []

        boosted = self._boost(fused, profile["boost"])
        top = sorted(boosted.items(), key=lambda kv: -kv[1])[:COARSE_TOPK]
        rescored = self._rescore(top, query_vec)

        if plan.rerank:
            cands = [(cid, s, self.store.get_chunk(cid)["text"])
                     for cid, s in rescored[:RERANK_TOPK]]
            rescored = self.reranker.rerank(query, cands) + rescored[RERANK_TOPK:]

        return [self._expand(cid, score) for cid, score in rescored[:top_k]]

    def _boost(self, fused: dict[int, float], boost_w: dict) -> dict[int, float]:
        """Post-fusion multiplicative recency/importance boost (plan §9 step 3)."""
        now = time.time()
        half_life = RECENCY_HALF_LIFE_DAYS * 86400.0
        out: dict[int, float] = {}
        for cid, score in fused.items():
            row = self.store.get_chunk(cid)
            if row is None:
                continue
            age = max(0.0, now - row["created_at"])
            recency = math.pow(0.5, age / half_life)
            factor = (1.0 + boost_w.get("recency", 0.0) * recency
                      + boost_w.get("importance", 0.0) * (row["importance"] or 0.0))
            out[cid] = score * factor
        return out

    def _rescore(self, top: list[tuple[int, float]],
                 query_vec: np.ndarray) -> list[tuple[int, float]]:
        """Exact float[768] cosine over the fused top-K; blended with RRF."""
        if not top:
            return []
        max_rrf = max(s for _, s in top) or 1.0
        out = []
        for cid, rrf_score in top:
            row = self.store.get_chunk(cid)
            if row is None:
                continue
            cos = 0.0
            if row["emb_full"]:
                emb = np.frombuffer(row["emb_full"], dtype=np.float32)
                if emb.shape[0] == DIM_FULL:
                    cos = float(np.dot(emb, query_vec))
            out.append((cid, 0.5 * (rrf_score / max_rrf) + 0.5 * max(0.0, cos)))
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
            neighbors=[n["text"] for n in self.store.neighbor_chunks(row)],
            episodes=episodes,
            meta={k: row[k] for k in ("meeting_id", "repo", "file_path", "page",
                                      "document_title", "speaker_span", "created_at")
                  if row[k] is not None})

    # -------------------------------------------------------------- answer

    def ask(self, query: str, top_k: int = ANSWER_TOPK) -> dict:
        """Route -> retrieve -> synthesize (Ollama) with citations.

        Returns {answer, plan, contexts}. Falls back to returning contexts
        verbatim when no LLM is reachable.
        """
        query_vec = self.embedder.embed_query(query)
        plan = self.router.route(query, query_vec)
        if plan.command:
            return {"answer": None, "plan": plan, "contexts": [],
                    "command": plan.command}
        contexts = self.retrieve(query, plan, top_k=top_k)
        if not contexts:
            return {"answer": "NOTFOUND — no matching memories.",
                    "plan": plan, "contexts": []}
        answer = self._synthesize(query, contexts)
        return {"answer": answer, "plan": plan, "contexts": contexts}

    def _synthesize(self, query: str, contexts: list[RetrievedContext]) -> str:
        blocks = []
        for c in contexts:
            src = c.title or c.source_type
            body = c.text
            if c.episodes:  # verbatim, speaker-attributed quotes for meetings
                body = "\n".join(f'{e["speaker"]}: {e["text"]}' for e in c.episodes)
            blocks.append(f"--- {c.citation()} ({c.source_type}: {src})\n{body}")
        context_str = "\n".join(blocks)
        prompt = (
            "You are Recall, an on-device memory assistant. Answer the question "
            "using ONLY the memory excerpts below. Cite the [id:chunk] markers "
            "of the excerpts you used. If the excerpts do not contain the "
            "answer, reply exactly NOTFOUND.\n\n"
            f"MEMORY EXCERPTS:\n{context_str}\n\nQUESTION: {query}\nANSWER:")
        try:
            import requests
            r = requests.post(
                f"{self.cfg.ollama_url.rstrip('/')}/api/chat",
                json={"model": self.cfg.llm_model, "stream": False,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=180)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception:
            # Offline fallback: hand back the raw evidence.
            return "LLM unavailable — top matches:\n" + "\n".join(
                f"{c.citation()} {c.text[:160]}" for c in contexts)
