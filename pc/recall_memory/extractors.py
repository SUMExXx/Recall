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

    Despite the parameter name, the caller (consolidate.job_llm_relations)
    passes a whole chunk — several sentences, not one — so the prompt asks for
    the single most important relation in that TEXT rather than implying a
    single-sentence input; the old wording invited the model to glue two
    unrelated fragments together with "and"/"for" as a fake "predicate".

    Returns None when no LLM is available — the cheap extractors already handled
    the confident cases, so this is a no-op on the offline/hash backend.
    """
    if llm is None or not getattr(llm, "available", False):
        return None
    import json

    from .llm_validate import is_valid_relation
    from .tracing import step
    prompt = (
        "Read the TEXT below and extract the SINGLE most important factual "
        "relationship in it as one (subject, predicate, object) triple. The "
        "predicate must be a real verb phrase (e.g. \"is\", \"has\", "
        "\"decided to\", \"will attend\", \"works at\") — NEVER a bare "
        "conjunction or preposition like \"and\"/\"for\"/\"of\". If the text "
        "has no clear relation, use predicate \"none\".\n"
        'Return ONLY JSON: {"subject":"","predicate":"","object":""}\n\n'
        f"TEXT: {sentence}")
    with step("extract_relation:llm_fallback",
             model=getattr(llm, "model", llm.name)) as s:
        s.detail(prompt=prompt)
        try:
            out = llm.generate(prompt, json=True, timeout=60)
            s.detail(response=out)
            d = json.loads(out)
        except Exception as e:
            s.detail(error=str(e))
            return None
        if not d.get("subject") or not d.get("object"):
            return None
        predicate = d.get("predicate") or "related_to"
        if predicate.strip().lower() == "none":
            s.detail(rejected="model reported no clear relation")
            return None
        if not is_valid_relation(d["subject"], d["object"], sentence, predicate):
            s.detail(rejected="weak/conjunction predicate, or object shares no "
                             "vocabulary with the source text — likely noise "
                             "or hallucinated")
            return None
    return {"subject": d["subject"], "predicate": predicate, "object": d["object"]}


def llm_extract_entities(text: str, llm=None) -> list[dict] | None:
    """Escalation for the entities regex can't see: multi-word names regex
    fragments (e.g. "Royal Enfield" + "GT" as two hits instead of one "Royal
    Enfield GT"), and lowercase topical phrases regex-by-design never catches
    at all ("office party", "flight to Noida") — these are exactly the shared
    vocabulary that would let genuinely-related memories cluster/link, and
    their absence is why cross-memory correlation stayed weak (see
    [[pipeline-audit]]). Every candidate is validated as a real (case-
    insensitive) substring of `text` before being trusted, so a model
    inventing an entity that never appears in the source is silently dropped,
    not written into the graph. Fragmented regex hits for the same name aren't
    deleted here — job_entity_resolution's substring/fuzzy merge (consolidate.py)
    already converges "royalenfield" into "royalenfieldgt" on its own.

    Returns None when no LLM is available."""
    if llm is None or not getattr(llm, "available", False):
        return None
    import json

    from .tracing import step
    prompt = (
        "Extract the concrete named entities and short topic phrases in this "
        "TEXT: people, places, organizations, products/models, and 2-4 word "
        "event/topic phrases a person would search for later. Skip generic "
        "words (\"today\", \"thing\"). Use each entity's EXACT wording as it "
        "appears in the TEXT — do not paraphrase or normalize it.\n"
        'Return ONLY JSON: {"entities":[{"name":"","type":"person|place|org|'
        'product|event|topic"}]}\n\n'
        f"TEXT: {text[:1000]}")
    with step("extract_entities:llm", model=getattr(llm, "model", llm.name)) as s:
        s.detail(prompt=prompt)
        try:
            raw = llm.generate(prompt, json=True, timeout=30.0)
            s.detail(response=raw)
            data = json.loads(raw)
        except Exception as e:
            s.detail(error=str(e))
            return None
        low = text.lower()
        out: list[dict] = []
        seen_spans: list[tuple[int, int]] = []
        for item in data.get("entities", []) if isinstance(data, dict) else []:
            name = (item.get("name") or "").strip()
            if len(name) < 3:
                continue
            idx = low.find(name.lower())
            if idx < 0:   # not actually in the source text -> hallucinated
                continue
            end = idx + len(name)
            if any(idx >= s0 and end <= e0 for s0, e0 in seen_spans):
                continue   # subsumed by an already-accepted (usually fuller) span
            seen_spans.append((idx, end))
            out.append({"name": text[idx:end],   # original casing from the text
                        "type": item.get("type") or "topic",
                        "start": idx, "end": end})
        s.detail(entities=len(out))
        return out
