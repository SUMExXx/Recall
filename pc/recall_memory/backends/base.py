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

from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import RecallConfig


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
