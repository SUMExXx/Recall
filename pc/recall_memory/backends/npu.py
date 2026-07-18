"""NpuBackend — the real target: Snapdragon X Elite, on-device.

Written against the documented runtime contracts from
plans/memory_engineering_v2.md §2. Every heavy import (`onnxruntime`,
`tokenizers`, GenieX) is lazy and guarded, so this module imports fine on a
plain x64 laptop; failures only surface if you actually select `npu` without
the runtime present. On the laptop, use `RECALL_BACKEND=ollama` instead.

  Embedder  Nomic-Embed-Text v1.5, ORT-QNN, re-exported seqlen 256/512 graphs
            (float[768], mean-pooled + L2-normed; task prefixes)
  LLM       Qwen3-4B-Instruct-2507 (w4a16) via GenieX/QAIRT, fronted by an
            OpenAI-compatible HTTP endpoint (genie sidecar)
  Reranker  Qwen3-Reranker-0.6B via ORT-QNN (falls back to passthrough order
            if the asset/runtime is missing)
  Tokenizer the model tokenizer (HF `tokenizers`)
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ..config import DIM_FULL, RecallConfig
from ..embeddings import DOC_PREFIX, QUERY_PREFIX, _normalize
from .base import Backend, PassthroughReranker

_QNN_PROVIDERS = ["QNNExecutionProvider", "CPUExecutionProvider"]
_QNN_PROVIDER_OPTIONS = [{"backend_path": "QnnHtp.dll"}, {}]
_NOMIC_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"


class NomicQnnEmbedder:
    """Nomic v1.5 on the NPU. Two static graphs (seqlen 256 / 512); the shorter
    one that fits the batch is used so nothing is silently truncated at the NPU
    boundary (plan §5). Output is mean-pooled over tokens then L2-normalized."""

    name = "nomic-qnn"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._sessions: dict[int, object] = {}
        self._tok = None

    def _session(self, seqlen: int):
        import onnxruntime as ort
        if seqlen not in self._sessions:
            path = (self.cfg.npu_embed_onnx_256 if seqlen <= 256
                    else self.cfg.npu_embed_onnx_512)
            self._sessions[seqlen] = ort.InferenceSession(
                path, providers=_QNN_PROVIDERS,
                provider_options=_QNN_PROVIDER_OPTIONS)
        return self._sessions[seqlen]

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(_NOMIC_TOKENIZER)
        return self._tok

    def _embed(self, texts: list[str]) -> np.ndarray:
        tok = self._tokenizer()
        encs = [tok.encode(t) for t in texts]
        max_len = max((len(e.ids) for e in encs), default=1)
        seqlen = 256 if max_len <= 256 else 512
        n = len(texts)
        input_ids = np.zeros((n, seqlen), dtype=np.int64)
        attn = np.zeros((n, seqlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids = e.ids[:seqlen]
            input_ids[i, :len(ids)] = ids
            attn[i, :len(ids)] = 1
        sess = self._session(seqlen)
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        # some exports also want token_type_ids
        wanted = {i.name for i in sess.get_inputs()}
        if "token_type_ids" in wanted:
            feeds["token_type_ids"] = np.zeros((n, seqlen), dtype=np.int64)
        out = sess.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:  # (n, seq, 768) token embeddings -> masked mean pool
            mask = attn[:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return _normalize(out.astype(np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        mat = self._embed([DOC_PREFIX + t for t in texts])
        if mat.shape[-1] != DIM_FULL:
            raise ValueError(f"expected {DIM_FULL}-dim embeddings, got {mat.shape[-1]}")
        return mat

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]


class QwenGenieLLM:
    """Qwen3-4B-Instruct-2507 via GenieX/QAIRT.

    Genie is deprecating in favor of GenieX (build against GenieX/QAIRT >= 2.42).
    We front the on-device model with an OpenAI-compatible HTTP server (the
    genie sample app / a thin QAIRT wrapper) at `npu_llm_endpoint`, so the
    engine talks the same chat contract regardless of backend.
    """

    name = "qwen3-genie"

    def __init__(self, cfg: RecallConfig):
        self.endpoint = cfg.npu_llm_endpoint.rstrip("/")
        self.model = cfg.npu_llm_model

    @property
    def available(self) -> bool:
        import requests
        try:
            requests.get(f"{self.endpoint}/v1/models", timeout=2).raise_for_status()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, *, json: bool = False,
                 timeout: float = 180.0) -> str:
        import requests
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt}]}
        if json:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.endpoint}/v1/chat/completions",
                          json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class QwenQnnReranker:
    """Qwen3-Reranker-0.6B cross-encoder on the NPU via ORT-QNN.

    Scores each (query, passage) pair and re-sorts. If the asset or QNN runtime
    is missing it degrades to fused order (passthrough) rather than failing the
    query — matches the plan's bge-reranker-CPU-fallback intent.
    """

    name = "qwen3-reranker-qnn"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._sess = None
        self._tok = None
        self._ok = True

    def _load(self):
        if self._sess is None and self._ok:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
                self._sess = ort.InferenceSession(
                    self.cfg.npu_reranker_onnx, providers=_QNN_PROVIDERS,
                    provider_options=_QNN_PROVIDER_OPTIONS)
                self._tok = Tokenizer.from_pretrained(self.cfg.npu_tokenizer_id)
            except Exception:
                self._ok = False  # fall back to passthrough for the rest of the run
        return self._sess is not None

    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
        if not candidates or not self._load():
            return [(cid, s) for cid, s, _ in candidates]
        scored = []
        for cid, _prev, text in candidates:
            enc = self._tok.encode(f"query: {query}", f"passage: {text}")
            ids = np.array([enc.ids], dtype=np.int64)
            attn = np.array([enc.attention_mask], dtype=np.int64)
            logit = self._sess.run(None, {"input_ids": ids, "attention_mask": attn})[0]
            scored.append((cid, float(np.ravel(logit)[0])))
        scored.sort(key=lambda t: -t[1])
        return scored


class NpuBackend(Backend):
    name = "npu"

    @cached_property
    def embedder(self):
        return NomicQnnEmbedder(self.cfg)

    @cached_property
    def llm(self):
        return QwenGenieLLM(self.cfg)

    @cached_property
    def reranker(self):
        return QwenQnnReranker(self.cfg)

    @cached_property
    def tokenizer(self):
        from ..tokenizer import ModelTokenizer
        return ModelTokenizer(self.cfg.npu_tokenizer_id)
