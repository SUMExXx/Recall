"""Tokenizers with char-span tracking (plan §5).

Every tokenizer returns `Token`s carrying exact `char_start`/`char_end` offsets
into the source text, so chunk windows keep byte-accurate provenance into
`NMO.content` (the chunker only depends on this interface).

Two implementations behind the backend switch:

  RegexTokenizer  word/punctuation regex — deterministic, dependency-free.
                  The offline `hash` backend and the ultimate fallback.
  ModelTokenizer  the real model tokenizer (HuggingFace `tokenizers`) with
                  `offsets` → char spans. Used by the `ollama` and `npu`
                  backends so the 256/400-token windows line up exactly with
                  the Nomic seqlen-256/512 graphs on the NPU.
"""
from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive


def tokenize(text: str) -> list[Token]:
    """Regex tokenization — the deterministic default."""
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))


class RegexTokenizer:
    """Deterministic, offline. Backs the `hash` test backend and the fallback."""

    name = "regex"

    def tokenize(self, text: str) -> list[Token]:
        return tokenize(text)

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)


class ModelTokenizer:
    """Real model tokenizer via HuggingFace `tokenizers`, char-offset aware.

    `spec` is either a local `tokenizer.json` path (shipped on the event PC) or
    a HuggingFace repo id (downloaded + cached on first use). If neither loads
    — no `tokenizers`, no network, bad id — it degrades to `RegexTokenizer` so
    ingestion never hard-fails on the token counter alone.
    """

    def __init__(self, spec: str):
        self.name = f"model:{spec}"
        self._tk = None
        self._fallback = RegexTokenizer()
        try:
            from tokenizers import Tokenizer as HFTokenizer
            if os.path.exists(spec):
                self._tk = HFTokenizer.from_file(spec)
            else:
                self._tk = HFTokenizer.from_pretrained(spec)
        except Exception as e:  # missing dep / offline / bad id
            warnings.warn(
                f"ModelTokenizer({spec!r}) unavailable ({e}); using regex fallback")
            self.name = f"regex(fallback:{spec})"

    def tokenize(self, text: str) -> list[Token]:
        if self._tk is None:
            return self._fallback.tokenize(text)
        enc = self._tk.encode(text)
        out: list[Token] = []
        for (start, end) in enc.offsets:
            if end > start:  # drop zero-width special tokens
                out.append(Token(text[start:end], start, end))
        return out

    def count_tokens(self, text: str) -> int:
        if self._tk is None:
            return self._fallback.count_tokens(text)
        return len(self._tk.encode(text).ids)
