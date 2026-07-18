"""Vector math + the offline HashEmbedder (plan §2, §9).

The two-tier Matryoshka scheme lives here because it is backend-independent:
full tier float32[768] (rescore, stored as BLOB on chunks); coarse tier
Matryoshka-truncate 768->256, renormalize, scalar-quantize to int8 (the hot
brute-force scan in vec_chunks).

Concrete embedders live in `recall_memory/backends/`:
  ollama.OllamaEmbedder   dev substitute (nomic-embed-text, 768-dim)
  npu.NomicQnnEmbedder    Nomic-Embed-Text v1.5 on the NPU via ORT-QNN
HashEmbedder (here) is the deterministic offline stub used by the `hash`
backend — lexical, not semantic, but stable across runs for tests/CI.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .config import DIM_COARSE, DIM_FULL, RecallConfig

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def matryoshka_coarse(full: np.ndarray) -> np.ndarray:
    """float32[..,768] -> int8[..,256]: truncate, renormalize, quantize."""
    trunc = _normalize(full[..., :DIM_COARSE].astype(np.float32))
    return np.clip(np.round(trunc * 127.0), -127, 127).astype(np.int8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


class HashEmbedder:
    """Deterministic bag-of-words hashing into DIM_FULL buckets.

    Not semantic — lexical overlap only — but stable across runs, which is what
    the test suite and offline demos need.
    """

    name = "hash"

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM_FULL), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int.from_bytes(hashlib.md5(word.encode()).digest()[:8], "little")
                out[i, h % DIM_FULL] += 1.0 if (h >> 63) == 0 else -1.0
        return _normalize(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def get_embedder(cfg: RecallConfig):
    """Backward-compatible shim — resolves the embedder from the active backend."""
    from .backends import get_backend
    return get_backend(cfg).embedder
