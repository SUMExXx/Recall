"""Cheap-extractors-first enrichment — plan §6.

Regex/keyword extraction handles the bulk of entities, decisions, and action
items; `llm_extract_relation` is the escalation hook for ambiguous sentences
(disabled by default — wire to Qwen3-4B / llama3.2 when NPU budget allows).
"""
from __future__ import annotations

import re

# Code-ish identifiers: CamelCase, snake_case, dotted.paths, UPPER_CONSTANTS
_CODE_ID_RE = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]+"   # dotted.path
    r"|[a-z]+_[a-z0-9_]+"                                      # snake_case
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"                      # CamelCase
    r"|[A-Z]{2,}[A-Z0-9_]*)\b")                                # ACRONYM/CONST

# Capitalized word runs mid-sentence (crude proper-noun catcher)
_PROPER_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_STOPWORDS = {
    "The", "This", "That", "These", "Those", "We", "They", "He", "She", "It",
    "And", "But", "For", "Not", "You", "Our", "His", "Her", "Its", "Their",
    "What", "When", "Who", "Why", "How", "Yes", "No", "Ok", "Okay", "So",
    "Then", "Also", "Just", "Let", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
}

_DECISION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwe(?:'ve| have)? decided (?:to|on|that) (?P<obj>[^.!?\n]+)",
        r"\bdecision\s*:\s*(?P<obj>[^.!?\n]+)",
        r"\blet'?s go with (?P<obj>[^.!?\n]+)",
        r"\bwe(?:'ll| will) (?:use|go with|adopt) (?P<obj>[^.!?\n]+)",
        r"\bagreed (?:to|on) (?P<obj>[^.!?\n]+)",
    )
]

_ACTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\baction item\s*:?\s*(?P<obj>[^.!?\n]+)",
        r"\b(?P<who>[A-Z][a-z]+) (?:will|is going to|needs to|should) (?P<obj>[^.!?\n]+)",
    )
]


def extract_entities(text: str) -> list[dict]:
    """Return [{name, type, start, end}] mentions, deduped by (name, start)."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for m in _CODE_ID_RE.finditer(text):
        key = (m.group(0), m.start())
        if key not in seen:
            seen.add(key)
            out.append({"name": m.group(0), "type": "identifier",
                        "start": m.start(), "end": m.end()})
    for m in _PROPER_RE.finditer(text):
        name = m.group(1)
        if name in _STOPWORDS or len(name) < 3:
            continue
        key = (name, m.start(1))
        if key not in seen:
            seen.add(key)
            out.append({"name": name, "type": "proper_noun",
                        "start": m.start(1), "end": m.end(1)})
    return out


def extract_decisions(text: str, speaker: str = "") -> list[dict]:
    """Return [{kind: decision|action_item, subject, object, sentence}]."""
    out: list[dict] = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(text):
            out.append({"kind": "decision", "subject": speaker or "team",
                        "predicate": "decided",
                        "object": m.group("obj").strip(),
                        "sentence": m.group(0).strip()})
    for pat in _ACTION_PATTERNS:
        for m in pat.finditer(text):
            subject = (m.groupdict().get("who") or speaker or "team")
            out.append({"kind": "action_item", "subject": subject,
                        "predicate": "will_do",
                        "object": m.group("obj").strip(),
                        "sentence": m.group(0).strip()})
    return out


def llm_extract_relation(sentence: str, llm=None) -> dict | None:
    """Escalation hook for ambiguous spans (plan §6): one (subject, predicate,
    object) triple via the backend LLM (Qwen3-4B on the NPU, llama3.2 on Ollama).

    Returns None when no LLM is available — the cheap extractors already handled
    the confident cases, so this is a no-op on the offline/hash backend.
    """
    if llm is None or not getattr(llm, "available", False):
        return None
    import json
    try:
        out = llm.generate(
            "Extract exactly one (subject, predicate, object) triple from the "
            'sentence as JSON: {"subject":"","predicate":"","object":""}. '
            f"Sentence: {sentence}", json=True, timeout=60)
        d = json.loads(out)
    except Exception:
        return None
    if d.get("subject") and d.get("object"):
        return {"subject": d["subject"],
                "predicate": d.get("predicate", "related_to"),
                "object": d["object"]}
    return None
