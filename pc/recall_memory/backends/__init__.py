"""Backend registry — the one place `RECALL_BACKEND` is resolved.

    get_backend(cfg).embedder / .llm / .reranker / .tokenizer

Backends are cheap to construct (providers are lazy); callers that share one
instance (the hub, the CLI) avoid loading a model more than once.
"""
from __future__ import annotations

from .. import tracing
from ..config import RecallConfig
from .base import (LLM, Backend, Embedder, NullLLM, PassthroughReranker,
                   Reranker, Tokenizer)


def get_backend(cfg: RecallConfig | None = None) -> Backend:
    cfg = cfg or RecallConfig()
    tracing.configure(cfg)   # single choke point every entry path passes through
    if cfg.backend == "hash":
        from .hash_backend import HashBackend
        return HashBackend(cfg)
    if cfg.backend == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend(cfg)
    if cfg.backend == "npu":
        from .npu import NpuBackend
        return NpuBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r} (npu | ollama | hash)")


__all__ = ["get_backend", "Backend", "Embedder", "LLM", "Reranker", "Tokenizer",
           "NullLLM", "PassthroughReranker"]
