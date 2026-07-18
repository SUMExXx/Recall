"""HashBackend — deterministic, offline, model-free. For tests and CI.

Lexical hash embeddings, no LLM (callers fall back to raw evidence), identity
reranker, regex tokenizer. Nothing here touches the network or a model file.
"""
from __future__ import annotations

from functools import cached_property

from .base import Backend, NullLLM, PassthroughReranker


class HashBackend(Backend):
    name = "hash"

    @cached_property
    def embedder(self):
        from ..embeddings import HashEmbedder
        return HashEmbedder()

    @cached_property
    def llm(self):
        return NullLLM()

    @cached_property
    def reranker(self):
        return PassthroughReranker()

    @cached_property
    def tokenizer(self):
        from ..tokenizer import RegexTokenizer
        return RegexTokenizer()
