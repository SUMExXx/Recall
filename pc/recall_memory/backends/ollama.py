"""OllamaBackend — local development substitute for the NPU stack.

Same model *family* and dimensions as the event PC, reached over Ollama's HTTP
API, so the exact same engine code runs against it. Selected with
`RECALL_BACKEND=ollama` on the dev laptop.

  Embedder  nomic-embed-text (768-dim; search_document:/search_query: prefixes)
  LLM       llama3.2:3b (chat, JSON mode)   — stands in for Qwen3-4B
  Reranker  passthrough (no cross-encoder locally)
  Tokenizer the real Qwen tokenizer (HF) so token windows match the NPU
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ..config import DIM_FULL, RecallConfig
from ..embeddings import DOC_PREFIX, QUERY_PREFIX, _normalize
from .base import Backend, PassthroughReranker


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, cfg: RecallConfig):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_embed_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self.url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        mat = np.array(resp.json()["embeddings"], dtype=np.float32)
        if mat.shape[-1] != DIM_FULL:
            raise ValueError(f"expected {DIM_FULL}-dim embeddings, got {mat.shape[-1]}")
        return _normalize(mat)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: RecallConfig):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_llm_model

    @property
    def available(self) -> bool:
        import requests
        try:
            requests.get(f"{self.url}/api/tags", timeout=2).raise_for_status()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, *, json: bool = False,
                 timeout: float = 180.0) -> str:
        import requests
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt}]}
        if json:
            body["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class OllamaBackend(Backend):
    name = "ollama"

    @cached_property
    def embedder(self):
        return OllamaEmbedder(self.cfg)

    @cached_property
    def llm(self):
        return OllamaLLM(self.cfg)

    @cached_property
    def reranker(self):
        return PassthroughReranker()

    @cached_property
    def tokenizer(self):
        from ..tokenizer import ModelTokenizer
        return ModelTokenizer(self.cfg.npu_tokenizer_id)
