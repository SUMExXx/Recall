"""Deterministic regex tokenizer with char-span tracking.

A lightweight stand-in for the model tokenizer: word/punctuation matches are
counted as tokens so chunk windows carry exact char_start/char_end provenance
into NMO.content. Swap `tokenize` for the real BPE tokenizer on the event PC if
token-budget precision matters; the chunker only depends on this interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))
