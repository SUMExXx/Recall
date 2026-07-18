"""Proactive Recall Engine (§4 CORE): rolling 45 s transcript window, embed,
retrieve against stored memories, push a recall card when a strong-enough match
surfaces — with a cooldown so it never spams mid-conversation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

WINDOW_SECONDS = 45.0
COOLDOWN_SECONDS = 60.0
MIN_SCORE = 0.45          # blended retrieval score threshold
MIN_WINDOW_CHARS = 40     # don't fire on two words


@dataclass
class _WindowItem:
    at: float
    text: str


@dataclass
class ProactiveRecallEngine:
    retriever: object                       # recall_memory.Retriever
    threshold: float = MIN_SCORE
    cooldown: float = COOLDOWN_SECONDS
    _window: list = field(default_factory=list)
    _last_fired: float = 0.0

    def observe(self, text: str, exclude_meeting_id: str | None = None,
                now: float | None = None) -> dict | None:
        """Feed one live utterance; returns a recall card dict or None."""
        now = now or time.time()
        self._window.append(_WindowItem(now, text))
        self._window = [w for w in self._window
                        if now - w.at <= WINDOW_SECONDS]
        if now - self._last_fired < self.cooldown:
            return None
        window_text = " ".join(w.text for w in self._window)
        if len(window_text) < MIN_WINDOW_CHARS:
            return None
        contexts = self.retriever.retrieve(window_text, top_k=3)
        contexts = [c for c in contexts
                    if not (exclude_meeting_id
                            and c.meta.get("meeting_id") == exclude_meeting_id)]
        if not contexts or contexts[0].score < self.threshold:
            return None
        self._last_fired = now
        best = contexts[0]
        return {
            "memory_id": best.memory_id, "chunk_id": best.chunk_id,
            "score": round(best.score, 3), "title": best.title,
            "source_type": best.source_type, "snippet": best.text[:240],
            "citation": best.citation(), "window": window_text[-240:],
        }

    def reset(self):
        self._window.clear()
        self._last_fired = 0.0
