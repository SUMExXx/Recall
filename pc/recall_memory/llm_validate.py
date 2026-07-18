"""Output validation for local-LLM enrichment calls (summaries, OKF blurbs,
relation extraction — plan §12/§7/§6).

These call sites hand a small local model a narrow prompt and, until now,
used the answer verbatim. That is exactly how a refusal ("I'm a large
language model, I don't have personal memories...") or an off-task
completion ends up permanently written into `memories.summary`, an OKF
manifest, or the relation graph — a write-time failure with no way to
recover the correct answer later, since the raw transcript isn't
re-consulted once the (bad) enrichment is persisted.

This is a cheap gate, not a quality guarantee: catch the obviously-wrong
shapes — a refusal/disclaimer, an empty-or-too-short answer, a response that
shares no vocabulary at all with what it was asked to summarize/extract from
— and let the caller fall back to the extractive text instead of persisting
garbage. It will not catch a fluent-but-wrong answer; that needs a stronger
model or a second verification pass, not a regex.
"""
from __future__ import annotations

import re

_REFUSAL_PATTERNS = re.compile(
    r"\b(i'?m (?:a|an) (?:large )?language model|as an ai(?:\s+language\s+model)?|"
    r"i don'?t have (?:access|personal|the ability)|i cannot|"
    r"i can'?t (?:provide|access|recall|help)|please provide|"
    r"no (?:specific )?text (?:was|is) provided|i'?m not able to|"
    r"i'?m sorry,? (?:but )?i)\b", re.IGNORECASE)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "as", "at",
    "by", "from", "we", "our", "i", "you", "your", "not", "no", "will",
    "have", "has", "had", "do", "does", "did", "so", "if", "then",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOPWORDS and len(w) > 2}


def is_valid_llm_summary(response: str, facts: str, min_chars: int = 15,
                         min_overlap: int = 1) -> bool:
    """A summary is acceptable only if it (a) isn't a refusal/disclaimer,
    (b) clears a minimum length, and (c) shares at least `min_overlap`
    content words with the facts it was asked to summarize — catches
    off-topic completions (a generic "Memory Retrieval Overview" that never
    actually mentions anything from the meeting)."""
    text = (response or "").strip()
    if len(text) < min_chars:
        return False
    if _REFUSAL_PATTERNS.search(text):
        return False
    return len(_content_words(text) & _content_words(facts)) >= min_overlap


def is_valid_relation(subject: str, object_: str, source_text: str) -> bool:
    """A (subject, predicate, object) triple is acceptable only if the
    object shares vocabulary with the sentence it was extracted from —
    catches triples hallucinated from unrelated context (e.g. a Whisper
    hallucination loop bleeding into the same chunk). An object with no
    content words of its own (too short/generic to judge) is let through
    rather than false-rejected."""
    if not subject or not object_:
        return False
    obj_words = _content_words(object_)
    if not obj_words:
        return True
    return bool(obj_words & _content_words(source_text))
