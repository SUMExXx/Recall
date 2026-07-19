"""Backend abstraction — the four model roles the engine depends on.

A `Backend` bundles an Embedder, an LLM, a Reranker, and a Tokenizer. Providers
are lazily constructed and cached, so selecting a backend (even `npu` on a
laptop that has no QNN runtime) never loads a model until something is used.

This is the single seam the `RECALL_BACKEND` switch turns:
  hash    -> HashBackend    (deterministic, offline; tests/CI)
  ollama  -> OllamaBackend  (local dev via Ollama)
  npu     -> NpuBackend     (Snapdragon X Elite; ORT-QNN + GenieX)
"""
from __future__ import annotations

import json
import logging
import re
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import RecallConfig

log = logging.getLogger("recall.backends")


@runtime_checkable
class Embedder(Protocol):
    name: str
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...   # (n, 768) L2-normed
    def embed_query(self, text: str) -> np.ndarray: ...              # (768,) L2-normed


@runtime_checkable
class Tokenizer(Protocol):
    name: str
    def tokenize(self, text: str) -> list: ...      # list[Token] with char spans
    def count_tokens(self, text: str) -> int: ...


@runtime_checkable
class LLM(Protocol):
    name: str
    @property
    def available(self) -> bool: ...
    def generate(self, prompt: str, *, json: bool = False,
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        """`schema`, when given, requests JSON-schema-constrained decoding
        (Ollama's grammar-constrained `format`) — the model is prevented from
        emitting anything that doesn't match, including any `enum` fields.
        Prefer this over `json=True` whenever the shape is known: it turns
        "the model echoed the placeholder back" into a class of error that
        can no longer happen, rather than one you detect-and-retry."""
        ...
    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        """Same contract as generate() but yields text deltas as they're
        produced instead of returning the full completion at once — no
        schema/json mode (those need the whole object to parse)."""
        ...


@runtime_checkable
class Reranker(Protocol):
    name: str
    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]: ...


class NullLLM:
    """No local LLM (hash backend / offline). Callers fall back to raw evidence."""

    name = "null"
    available = False

    def generate(self, prompt: str, *, json: bool = False,
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        raise RuntimeError("no LLM configured on this backend")

    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        raise RuntimeError("no LLM configured on this backend")
        yield   # pragma: no cover — makes this a generator function


class PassthroughReranker:
    """Identity reranker — keeps the fused order. Backs the hash + ollama
    backends (no cross-encoder on the dev laptop); NPU swaps in Qwen3-Reranker."""

    name = "passthrough"

    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
        return [(cid, score) for cid, score, _text in candidates]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LlmReranker:
    """Cross-encoder-quality reranking via the on-device LLM (RankGPT-style
    listwise scoring in ONE call).

    The dedicated cross-encoder slot (Qwen3-Reranker ONNX / bge-reranker on
    sentence-transformers) needs an asset or a torch stack we don't have on the
    Snapdragon. Rather than leave the rerank step a no-op passthrough — which is
    what the trace showed (`rerank 0.0 ms`, fused order untouched) — this asks
    the LLM already loaded on the NPU to score each candidate's relevance to the
    query on a 0..1 scale. That is a genuine (query, passage) judgment, so the
    relevance-cutoff downstream finally has real scores to gate on.

    One extra LLM call per query. Skips itself for <=1 candidate, and on ANY
    parse/timeout failure returns the fused order unchanged so a query never
    breaks because reranking hiccuped."""

    name = "llm-listwise"

    def __init__(self, llm, cfg: RecallConfig | None = None, max_text: int = 400):
        self.llm = llm
        self.max_text = max_text

    _PROMPT = (
        "You are a relevance judge for a personal-memory search engine. For "
        "each PASSAGE, score how directly it ANSWERS the QUERY:\n"
        "  1.0 = states the specific answer to the question\n"
        "  0.5 = same general subject but does NOT actually answer it\n"
        "  0.0 = unrelated\n"
        "Judge meaning, not keyword overlap, and do not reward a passage just "
        "for being about a similar subject (a note about a bike is NOT a good "
        "answer to a question about a project). Return ONLY JSON, no prose:\n"
        '{{"results":[{{"id":<passage number>,"score":<0.0-1.0>}}, ...]}}\n'
        "one entry per passage.\n\nQUERY: {query}\n\nPASSAGES:\n{listing}")

    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
        base = [(cid, score) for cid, score, _text in candidates]
        if len(candidates) <= 1 or not getattr(self.llm, "available", False):
            return base
        listing = "\n".join(
            f"[{i}] {(text or '').strip()[:self.max_text]}"
            for i, (_cid, _sc, text) in enumerate(candidates))
        try:
            raw = self.llm.generate(
                self._PROMPT.format(query=query, listing=listing),
                json=True, timeout=30.0)
            m = _JSON_RE.search(raw or "")
            data = json.loads(m.group(0)) if m else {}
            scores: dict[int, float] = {}
            for item in data.get("results", []):
                idx = int(item["id"])
                if 0 <= idx < len(candidates):
                    scores[candidates[idx][0]] = max(0.0, min(1.0, float(item["score"])))
            if not scores:
                return base
            # candidates the judge omitted -> 0.0, so the cutoff can drop them
            ranked = sorted(((cid, scores.get(cid, 0.0)) for cid, _s, _t in candidates),
                            key=lambda t: -t[1])
            return ranked
        except Exception as e:
            log.debug("llm reranker fell back to fused order: %s", e)
            return base


class Backend:
    """Base bundle. Subclasses override the cached_property providers."""

    name = "base"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg

    @cached_property
    def embedder(self) -> Embedder:
        raise NotImplementedError

    @cached_property
    def llm(self) -> LLM:
        raise NotImplementedError

    @cached_property
    def reranker(self) -> Reranker:
        raise NotImplementedError

    @cached_property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError
