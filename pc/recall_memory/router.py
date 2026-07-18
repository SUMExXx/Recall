"""4-tier query router — plan §8.

Tier 0  command grammar (forget / bookmark / summarize)        <1 ms
Tier 1  rules + filter extraction (dates, speakers, entities)  <5 ms
Tier 2  embedding-prototype (reuses the retrieval query vec)   ~10 ms
Tier 3  LLM planner, plan JSON, validated w/ one retry         opt-in

All tiers emit the same RoutePlan (the unified needs{} struct); tier-3's JSON
is the superset. Tier 3 is OFF by default per plan recommendation #3.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field

import numpy as np

from .config import RecallConfig, WEIGHT_PROFILES
from .store import MemoryStore
from .tracing import step

QUERY_TYPES = ("meeting", "code", "decision", "timeline", "general")

_DEFAULT_NEEDS = {"bm25": True, "vector": True, "graph": True,
                  "metadata_filter": True, "entity_index": True}

# Below this word count, an LLM classification call buys nothing — "hi",
# "thanks", "ok" have no plan worth extracting — but still cost a full local
# LLM round-trip (measured: 5-16s) if tier-3 is on. Tiers 0/1 already handle
# every query with real structure (commands, dates, decision/code/timeline
# keywords); this only skips tier-3 for the trivial leftovers, straight to
# the ~2ms embedding-prototype tier instead.
_TIER3_MIN_WORDS = 3

# Tier-3 planner: a real JSON Schema (not a pipe-joined placeholder string) so
# constrained decoding can enforce query_type/path as an `enum` — see _tier3().
_TIER3_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": list(QUERY_TYPES)},
        "entities": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "object",
            "properties": {
                "date_range": {"type": ["array", "null"]},
                "source_type": {"type": ["string", "null"]},
                "speaker": {"type": ["string", "null"]},
            },
        },
        "needs": {
            "type": "object",
            "properties": {k: {"type": "boolean"} for k in _DEFAULT_NEEDS},
        },
        "path": {"type": "string", "enum": ["fast", "deep"]},
        "rerank": {"type": "boolean"},
    },
    "required": ["query_type", "path"],
}

_TIER3_EXAMPLE = json.dumps({
    "query_type": "decision", "entities": ["Redis"],
    "filters": {"date_range": None, "source_type": None, "speaker": None},
    "needs": {"bm25": True, "vector": False, "graph": True,
             "metadata_filter": True, "entity_index": True},
    "path": "deep", "rerank": True})


@dataclass
class RoutePlan:
    query_type: str = "general"
    entities: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)   # date_range, source_type, speaker, meeting_id, repo
    needs: dict = field(default_factory=lambda: dict(_DEFAULT_NEEDS))
    path: str = "deep"                            # fast | deep
    rerank: bool = True
    weight_profile: str = "general"
    tier: int = 1
    command: dict | None = None                   # tier-0 only


# --------------------------------------------------------------- tier 0

_FORGET_RE = re.compile(
    r"\bforget (?:the )?last (?P<n>\d+)\s*(?P<unit>minute|min|hour|hr)s?\b", re.I)
_SUMMARIZE_RE = re.compile(r"\bsummari[sz]e (?:the )?last meeting\b", re.I)
_BOOKMARK_RE = re.compile(r"^\s*bookmark\b", re.I)


def _tier0(query: str) -> RoutePlan | None:
    m = _FORGET_RE.search(query)
    if m:
        mins = int(m.group("n")) * (60 if m.group("unit").lower().startswith("h") else 1)
        return RoutePlan(query_type="meeting", tier=0, path="fast",
                         rerank=False,
                         needs={k: False for k in _DEFAULT_NEEDS},
                         command={"action": "forget_last", "minutes": mins})
    if _SUMMARIZE_RE.search(query):
        return RoutePlan(query_type="meeting", tier=0, path="fast", rerank=False,
                         filters={"source_type": "meeting"},
                         needs={**{k: False for k in _DEFAULT_NEEDS},
                                "metadata_filter": True},
                         weight_profile="decision",
                         command={"action": "summarize_last_meeting"})
    if _BOOKMARK_RE.search(query):
        return RoutePlan(tier=0, path="fast", rerank=False,
                         needs={k: False for k in _DEFAULT_NEEDS},
                         command={"action": "bookmark"})
    return None


# --------------------------------------------------------------- tier 1

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]


def _parse_dates(query: str, now: dt.datetime | None = None) -> tuple[int, int] | None:
    now = now or dt.datetime.now()
    q = query.lower()
    day = dt.timedelta(days=1)

    def span(d0: dt.datetime, days: float = 1) -> tuple[int, int]:
        start = d0.replace(hour=0, minute=0, second=0, microsecond=0)
        return (int(start.timestamp()), int((start + dt.timedelta(days=days)).timestamp()))

    if "today" in q:
        return span(now)
    if "yesterday" in q:
        return span(now - day)
    if "last week" in q:
        return span(now - dt.timedelta(days=7), 7)
    if "this week" in q:
        return span(now - dt.timedelta(days=now.weekday()), 7)
    if "last month" in q:
        return span(now - dt.timedelta(days=30), 30)
    m = re.search(r"last (\w+day)", q)
    if m and m.group(1) in _WEEKDAYS:
        target = _WEEKDAYS.index(m.group(1))
        delta = (now.weekday() - target) % 7 or 7
        return span(now - dt.timedelta(days=delta))
    return None


_CODE_HINT_RE = re.compile(
    r"\b(function|class|file|repo|code|implement|endpoint|bug|error|import)\b"
    r"|[a-z]+_[a-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+", )
_DECISION_HINT_RE = re.compile(
    r"\b(who|decided|decide|decision|suggested|agreed|chose|action item)\b", re.I)
_TIMELINE_HINT_RE = re.compile(
    r"\b(when|timeline|history of|originally|at first|before we|changed)\b", re.I)
_META_ONLY_RE = re.compile(
    r"\b(show|list)\b.*\b(meetings?|notes?|documents?|pdfs?)\b", re.I)
_SOURCE_HINTS = {"meeting": "meeting", "meetings": "meeting", "pdf": "pdf",
                 "document": "pdf", "repo": "github_repo", "code": "github_repo",
                 "whiteboard": "image", "photo": "image", "screenshot": "image"}


def _extract_entities(query: str, store: MemoryStore) -> list[str]:
    """Terms from the query that exist in the entity index (the cheap lookup)."""
    cands: set[str] = set()
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_.]{2,}\b", query):
        cands.add(m.group(0))
    # multiword proper nouns too
    for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
        cands.add(m.group(0))
    return [c for c in cands if store.entity_exists(c)]


def _tier1(query: str, store: MemoryStore) -> RoutePlan | None:
    filters: dict = {}
    dr = _parse_dates(query)
    if dr:
        filters["date_range"] = dr
    speakers = {s.lower(): s for s in store.known_speakers()}
    for word in re.findall(r"\b[A-Z][a-z]+\b", query):
        if word.lower() in speakers:
            filters["speaker"] = speakers[word.lower()]
            break
    entities = _extract_entities(query, store)

    # metadata-only listing ("show me meetings from last Friday")
    if _META_ONLY_RE.search(query):
        for word, st in _SOURCE_HINTS.items():
            if re.search(rf"\b{word}\b", query, re.I):
                filters["source_type"] = st
                break
        return RoutePlan(query_type="meeting", entities=entities, filters=filters,
                         path="fast", rerank=False, weight_profile="timeline",
                         needs={**{k: False for k in _DEFAULT_NEEDS},
                                "metadata_filter": True}, tier=1)

    if _DECISION_HINT_RE.search(query):
        return RoutePlan(query_type="decision", entities=entities, filters=filters,
                         weight_profile="decision",
                         needs={"bm25": True, "vector": False, "graph": True,
                                "metadata_filter": True, "entity_index": True},
                         tier=1)
    if _TIMELINE_HINT_RE.search(query):
        return RoutePlan(query_type="timeline", entities=entities, filters=filters,
                         weight_profile="timeline",
                         needs={"bm25": False, "vector": False, "graph": True,
                                "metadata_filter": True, "entity_index": True},
                         tier=1)
    if _CODE_HINT_RE.search(query):
        return RoutePlan(query_type="code", entities=entities, filters=filters,
                         weight_profile="code",
                         needs={"bm25": True, "vector": True, "graph": False,
                                "metadata_filter": bool(filters),
                                "entity_index": True}, tier=1)
    if filters or entities:
        return RoutePlan(query_type="general", entities=entities, filters=filters,
                         weight_profile="general", tier=1)
    return None  # fall through to tier 2


# --------------------------------------------------------------- tier 2

_PROTOTYPES = {
    "decision": ["who decided to use redis", "what did we decide about the api",
                 "who suggested websockets"],
    "code": ["how is the auth middleware implemented",
             "where is the login function defined"],
    "timeline": ["when did we switch databases",
                 "what did we originally plan for storage"],
    "general": ["explain our architecture", "how do we handle errors",
                "what do we know about caching"],
}


class Router:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 embedder=None, llm=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.embedder = embedder
        self.llm = llm
        self._proto_vecs: dict[str, np.ndarray] | None = None

    def _prototypes(self) -> dict[str, np.ndarray]:
        if self._proto_vecs is None:
            self._proto_vecs = {
                qt: np.mean(self.embedder.embed_documents(examples), axis=0)
                for qt, examples in _PROTOTYPES.items()}
        return self._proto_vecs

    def _tier2(self, query: str, query_vec: np.ndarray) -> RoutePlan:
        best_qt, best_sim = "general", -2.0
        for qt, proto in self._prototypes().items():
            sim = float(np.dot(query_vec, proto) /
                        (np.linalg.norm(query_vec) * np.linalg.norm(proto) + 1e-9))
            if sim > best_sim:
                best_qt, best_sim = qt, sim
        profile = best_qt if best_qt in WEIGHT_PROFILES else "general"
        return RoutePlan(query_type=best_qt, weight_profile=profile, tier=2)

    def _tier3(self, query: str) -> RoutePlan | None:
        """LLM planner: emits the plan JSON; validated, one retry, tier-1 fallback.

        Uses JSON-schema-constrained decoding (an `enum` for query_type/path,
        not a pipe-joined placeholder string embedded in the prompt) plus one
        concrete few-shot example. The earlier version showed the model the
        literal string "meeting|code|decision|timeline|general" as the VALUE
        to fill in — small models (llama3.2:3b) would parrot that placeholder
        back verbatim as their answer, fail validation, and burn a full retry
        (and ~20s) before landing on a real value, on every single tier-3
        call. A real JSON schema makes that class of failure structurally
        impossible on backends that honor it (Ollama); the enum check below
        stays as a safety net for backends where schema support is best-effort
        (GenieX/QAIRT — see QwenGenieLLM.generate).
        """
        if self.llm is None:
            return None
        prompt = (
            "Classify this memory-recall query and emit a retrieval plan as JSON.\n\n"
            f'Example — query "Who decided to use Redis?" ->\n{_TIER3_EXAMPLE}\n\n'
            f"Query: {query}\nJSON only.")
        for attempt in range(1, 3):
            with step("router:tier3_llm_planner", attempt=attempt,
                      model=getattr(self.llm, "model", self.llm.name)) as s:
                s.detail(prompt=prompt)
                try:
                    content = self.llm.generate(prompt, schema=_TIER3_SCHEMA,
                                                timeout=60)
                    s.detail(response=content)
                    plan = json.loads(content)
                    qt = plan.get("query_type", "general")
                    if qt not in QUERY_TYPES:
                        s.detail(rejected=f"unknown query_type {qt!r}")
                        continue
                    needs = {k: bool(plan.get("needs", {}).get(k, v))
                             for k, v in _DEFAULT_NEEDS.items()}
                    s.detail(accepted=True, query_type=qt)
                    return RoutePlan(
                        query_type=qt, entities=list(plan.get("entities") or []),
                        filters={k: v for k, v in (plan.get("filters") or {}).items() if v},
                        needs=needs, path=plan.get("path", "deep"),
                        rerank=bool(plan.get("rerank", True)),
                        weight_profile=qt if qt in WEIGHT_PROFILES else "general",
                        tier=3)
                except Exception as e:
                    s.detail(parse_error=str(e))
                    continue
        return None

    def route(self, query: str, query_vec: np.ndarray | None = None) -> RoutePlan:
        with step("router:tier0_command_grammar") as s:
            plan = _tier0(query)
            s.detail(matched=plan is not None,
                     command=plan.command if plan else None)
        if plan:
            return plan

        with step("router:tier1_rules_and_filters") as s:
            plan = _tier1(query, self.store)
            s.detail(matched=plan is not None,
                     query_type=plan.query_type if plan else None,
                     filters=plan.filters if plan else None)
        if plan:
            return plan

        if (self.cfg.use_tier3_planner and len(query.split()) >= _TIER3_MIN_WORDS
                and self.llm is not None and self.llm.available):
            plan = self._tier3(query)   # instruments its own step(s)
            if plan:
                return plan

        if self.embedder is not None and query_vec is not None:
            with step("router:tier2_embedding_prototype") as s:
                plan = self._tier2(query, query_vec)
                s.detail(query_type=plan.query_type,
                         weight_profile=plan.weight_profile)
            return plan
        return RoutePlan(tier=1)  # tier-1 default: general, everything on
