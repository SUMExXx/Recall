"""Query routing — flattened (plan §8's 4-tier router removed by request).

The tiered router (command grammar -> regex rules -> embedding-prototype ->
LLM planner) added real, measured latency for uneven benefit: tier-3 alone
cost 5-8s per call and, per production trace logs, fired on most free-form
questions (well past the plan's own ">20% -> tighten tier-1" guidance);
tier-0's "command" detection was never actually wired to execute anything
downstream (forget/bookmark run only through the WS `button` path); the
fast/deep `path` field only ever changed one redundant condition. None of
that complexity was buying its cost, so it's gone.

What every query gets now: ONE fixed plan — run bm25 + vector + graph +
entity_index, weighted-RRF fused under the "general" profile, always
reranked. Entity extraction is kept as a plain utility (regex candidate
terms checked against the entity index) — not a classification tier, just
wiring so the entity_index/graph routes have something to search instead of
always coming back empty.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import MemoryStore
from .tracing import step

_DEFAULT_NEEDS = {"bm25": True, "vector": True, "graph": True,
                  "metadata_filter": False, "entity_index": True}


@dataclass
class RoutePlan:
    query_type: str = "general"
    entities: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)   # reserved: date_range, source_type, speaker
    needs: dict = field(default_factory=lambda: dict(_DEFAULT_NEEDS))
    rerank: bool = True
    weight_profile: str = "general"


_ENTITY_CANDIDATE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.]{2,}\b")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _extract_entities(query: str, store: MemoryStore) -> list[str]:
    """Terms from the query that exist in the entity index (the cheap
    lookup) — feeds entity_index/graph so they aren't always empty."""
    cands: set[str] = set()
    for m in _ENTITY_CANDIDATE_RE.finditer(query):
        cands.add(m.group(0))
    for m in _PROPER_NOUN_RE.finditer(query):    # multiword proper nouns too
        cands.add(m.group(0))
    return [c for c in cands if store.entity_exists(c)]


class Router:
    """No tiers, no classification, no LLM call — every query gets the same
    fixed RoutePlan. Kept as a class (not a bare function) so existing
    Retriever/hub code that holds a `.router` reference needs no restructuring."""

    def __init__(self, store: MemoryStore, cfg=None, **_ignored):
        self.store = store
        self.cfg = cfg

    def route(self, query: str, query_vec=None) -> RoutePlan:
        with step("route:extract_entities") as s:
            entities = _extract_entities(query, self.store)
            s.detail(entities=entities)
        return RoutePlan(entities=entities)
