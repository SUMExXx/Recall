"""Proactive Recall Engine (§4 CORE): rolling 45 s transcript window, embed,
retrieve against stored memories, push a recall card when a strong-enough match
surfaces — with throttling so it never spams mid-conversation.

Two SEPARATE throttles, on purpose:
  fire cooldown       after a successful recall card fires, stay quiet for
                      `cooldown` seconds (the original behavior).
  attempt throttle     never even ATTEMPT another embed+retrieve within
                      `min_attempt_interval` seconds, fire or not.

The attempt throttle is the important one: `_last_fired` only ever moves when
a card successfully fires (score >= threshold). A dry spell — the store has
nothing to match yet, or every candidate scores low — never updates
`_last_fired`, so a cooldown keyed only on fires is fail-OPEN: it never
throttles anything until the very first success. In practice this meant every
single ASR-transcribed chunk (one call every few seconds of continuous
speech) re-embedded the whole rolling window and ran a full retrieve — dozens
of calls a minute with nothing ever gating them. `_last_attempt` is updated on
EVERY call regardless of outcome, so the attempt throttle is fail-SAFE.
"""
import threading
import time
from dataclasses import dataclass, field

WINDOW_SECONDS = 45.0
COOLDOWN_SECONDS = 60.0          # quiet period after a successful fire
MIN_ATTEMPT_INTERVAL = 8.0       # never retrieve more often than this, fire or not
MIN_SCORE = 0.45                 # blended retrieval score threshold
MIN_WINDOW_CHARS = 40            # don't fire on two words


@dataclass
class _WindowItem:
    at: float
    text: str


@dataclass
class ProactiveRecallEngine:
    retriever: object                       # recall_memory.Retriever
    threshold: float = MIN_SCORE
    cooldown: float = COOLDOWN_SECONDS
    min_attempt_interval: float = MIN_ATTEMPT_INTERVAL
    _window: list = field(default_factory=list)
    _last_fired: float = 0.0
    _last_attempt: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def observe(self, text: str, exclude_meeting_id: str | None = None,
                now: float | None = None) -> dict | None:
        """Feed one live utterance; returns a recall card dict or None."""
        now = now or time.time()
        with self._lock:
            self._window.append(_WindowItem(now, text))
            self._window = [w for w in self._window
                            if now - w.at <= WINDOW_SECONDS]
            if now - self._last_fired < self.cooldown:
                return None
            if now - self._last_attempt < self.min_attempt_interval:
                return None
            window_text = " ".join(w.text for w in self._window)
            if len(window_text) < MIN_WINDOW_CHARS:
                return None
            self._last_attempt = now   # set BEFORE the retrieve — throttles dry spells too

        contexts = self.retriever.retrieve(window_text, top_k=3)
        contexts = [c for c in contexts
                    if not (exclude_meeting_id
                            and c.meta.get("meeting_id") == exclude_meeting_id)]
        
        with self._lock:
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
        with self._lock:
            self._window.clear()
            self._last_fired = 0.0
            self._last_attempt = 0.0
