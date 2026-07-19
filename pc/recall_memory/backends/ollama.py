"""OllamaBackend — local development substitute for the NPU stack.

Same model *family* and dimensions as the event PC, reached over Ollama's HTTP
API, so the exact same engine code runs against it. Selected with
`RECALL_BACKEND=ollama` on the dev laptop.

  Embedder  nomic-embed-text (768-dim; search_document:/search_query: prefixes)
  LLM       llama3.2:3b (chat, JSON mode)   — stands in for Qwen3-4B
  Reranker  BAAI/bge-reranker-v2-m3 via sentence-transformers, GPU-accelerated
            if available, when RECALL_RERANKER_MODEL is set — else passthrough
            (Ollama itself has no rerank endpoint to fall back to)
  Tokenizer the real Qwen tokenizer (HF) so token windows match the NPU
"""
from __future__ import annotations

import json
import logging
import math
from functools import cached_property

import numpy as np

from ..config import DIM_FULL, RecallConfig
from ..embeddings import DOC_PREFIX, QUERY_PREFIX, _normalize
from ..tracing import step
from .base import Backend, PassthroughReranker

log = logging.getLogger("recall.backends.ollama")


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, cfg: RecallConfig):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_embed_model

    def _embed(self, texts: list[str]) -> np.ndarray:
        import requests
        resp = requests.post(
            f"{self.url}/api/embed",
            # keep_alive: keep the model resident — a cold reload adds seconds
            json={"model": self.model, "input": texts, "keep_alive": "30m"},
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
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        import requests
        body = {"model": self.model, "stream": False,
                "keep_alive": "30m",              # no cold reload between asks
                "options": {"num_predict": 320},  # cited answers, bounded latency
                # think=False: some Ollama models (e.g. "thinking"-capable
                # Gemma builds) spend their ENTIRE num_predict budget on
                # hidden reasoning tokens and stop at the cap before ever
                # emitting the actual answer — content comes back empty,
                # done_reason="length". Confirmed directly against gemma4:
                # with thinking on, a trivial tier-3 planner call burned all
                # 320 tokens reasoning about "hi" and returned "". Ignored by
                # models that don't support toggling it.
                "think": False,
                "messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            # Ollama grammar-constrains generation to this JSON schema —
            # including any `enum` fields — so a small model literally
            # cannot emit an out-of-set value (no more echoing a pipe-joined
            # placeholder like "meeting|code|decision|timeline|general" back
            # verbatim; that string isn't a legal value under the schema).
            body["format"] = schema
        elif json:
            body["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        """Like generate(), but yields text deltas as Ollama produces them
        (NDJSON — one `{"message": {"content": "..."}, "done": bool}` object
        per line) instead of waiting for the full completion. No schema/json
        mode here — those need the whole object to parse, which defeats
        streaming; use generate() for structured output."""
        import requests
        body = {"model": self.model, "stream": True, "keep_alive": "30m",
                "options": {"num_predict": 320}, "think": False,
                "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=timeout, stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            delta = obj.get("message", {}).get("content", "")
            if delta:
                yield delta
            if obj.get("done"):
                break


class LocalTransformersReranker:
    """Local CPU/GPU cross-encoder reranker using Hugging Face sentence-transformers.
    Selected when RECALL_RERANKER_MODEL is configured and sentence-transformers is installed.
    """
    name = "bge-reranker-local"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading %s onto %s (first call downloads the model)…",
                    self.model_name, device)
            self._model = CrossEncoder(self.model_name, device=device)
            log.info("%s ready on %s", self.model_name, device)
        return self._model

    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
        if not candidates:
            return []
        with step("rerank:cross_encoder", model=self.model_name,
                  candidates=len(candidates)) as s:
            model = self._load()
            pairs = [[query, text] for _, _, text in candidates]
            raw = model.predict(pairs)
            # bge-reranker-v2-m3 emits a single unbounded logit, not a 0-1
            # score — sigmoid-normalize it (the model card's own convention)
            # so retrieval.py's relevance_cutoff can threshold it meaningfully
            # instead of comparing raw, unnormalized logits against a magic
            # number.
            scored = [(candidates[i][0], 1.0 / (1.0 + math.exp(-float(raw[i]))))
                     for i in range(len(candidates))]
            scored.sort(key=lambda t: -t[1])
            s.detail(device=str(model.model.device))
        return scored


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
        import logging
        log = logging.getLogger("recall.backends.ollama")
        model_name = self.cfg.reranker_model
        if model_name:
            try:
                import sentence_transformers  # noqa: F401 — availability probe only
                log.info("loading cross-encoder reranker %s (sentence-transformers)…",
                        model_name)
                return LocalTransformersReranker(model_name)
            except ImportError:
                log.warning("RECALL_RERANKER_MODEL=%s is set but sentence-transformers "
                           "is not installed — falling back to passthrough. "
                           'Run: pip install -e ".[rerank]"', model_name)
        return PassthroughReranker()

    @cached_property
    def tokenizer(self):
        from ..tokenizer import ModelTokenizer
        return ModelTokenizer(self.cfg.npu_tokenizer_id)

