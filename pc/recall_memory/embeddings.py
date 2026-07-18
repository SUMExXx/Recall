"""Embedding providers + two-tier Matryoshka vectors (plan §2, §9).

Full tier: float32[768] (rescore, stored as BLOB on chunks).
Coarse tier: Matryoshka truncate 768->256, renormalize, scalar-quantize to int8
(hot brute-force scan in vec_chunks).

OllamaEmbedder is the dev substitute for the AI Hub Nomic v1.5 NPU asset — same
model family and dimension, including the search_document:/search_query: task
prefixes. HashEmbedder is a deterministic offline fallback (tests, no Ollama).
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


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, cfg: RecallConfig):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.embed_model

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


def get_embedder(cfg: RecallConfig):
    if cfg.embedder == "hash":
        return HashEmbedder()
    if cfg.embedder == "ollama":
        return OllamaEmbedder(cfg)
    # auto: probe Ollama, fall back to hash
    try:
        import requests
        requests.get(f"{cfg.ollama_url.rstrip('/')}/api/tags", timeout=2).raise_for_status()
        return OllamaEmbedder(cfg)
    except Exception:
        return HashEmbedder()
