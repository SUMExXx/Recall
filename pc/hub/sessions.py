"""Live meeting sessions: buffer utterances per capture device, handle the
bookmark / forget-last buttons against the live buffer, and hand the finished
meeting to the memory engine on meeting_end."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from recall_memory.nmo import new_id


@dataclass
class MeetingSession:
    meeting_id: str
    device_id: str
    capture_device: str
    title: str
    started_at: float
    utterances: list = field(default_factory=list)
    bookmarked: bool = False
    privacy_tag: str = "normal"

    def add_utterance(self, text: str, speaker: str = "", t_start: float | None = None,
                      t_end: float | None = None, asr_provider: str = "passthrough",
                      asr_confidence: float = 1.0, lang: str = "en") -> dict:
        now = time.time()
        u = {"speaker": speaker, "text": text,
             "t_start": t_start if t_start is not None else now,
             "t_end": t_end if t_end is not None else now,
             "asr_provider": asr_provider, "asr_confidence": asr_confidence,
             "lang": lang}
        self.utterances.append(u)
        return u

    def forget_last(self, minutes: float) -> int:
        """Drop buffered utterances from the last N minutes (live forget)."""
        cutoff = time.time() - minutes * 60.0
        before = len(self.utterances)
        self.utterances = [u for u in self.utterances if u["t_end"] < cutoff]
        return before - len(self.utterances)


class SessionManager:
    def __init__(self, ingestor, store):
        self.ingestor = ingestor
        self.store = store
        self.active: dict[str, MeetingSession] = {}   # device_id -> session

    def start(self, device_id: str, capture_device: str = "arduino",
              meeting_id: str | None = None, title: str | None = None,
              privacy_tag: str = "normal") -> MeetingSession:
        s = MeetingSession(
            meeting_id=meeting_id or f"mtg-{new_id()[:8]}",
            device_id=device_id, capture_device=capture_device,
            title=title or f"Meeting {time.strftime('%Y-%m-%d %H:%M')}",
            started_at=time.time(), privacy_tag=privacy_tag)
        self.active[device_id] = s
        return s

    def get(self, device_id: str) -> MeetingSession | None:
        return self.active.get(device_id)

    def bookmark(self, device_id: str) -> bool:
        s = self.active.get(device_id)
        if s:
            s.bookmarked = True
        return bool(s)

    def forget_last(self, device_id: str, minutes: float) -> dict:
        """Forget across the live buffer AND anything already persisted."""
        dropped_live = 0
        s = self.active.get(device_id)
        t_to = time.time()
        persisted = {"episodes": 0, "chunks": 0}
        if s:
            dropped_live = s.forget_last(minutes)
            persisted = self.store.forget_time_range(
                s.meeting_id, t_to - minutes * 60.0, t_to)
        return {"live_utterances": dropped_live, **persisted}

    def end(self, device_id: str) -> str | None:
        """Close the session and ingest it as a meeting NMO. Returns memory_id."""
        s = self.active.pop(device_id, None)
        if s is None or not s.utterances:
            return None
        memory_id = self.ingestor.ingest_meeting({
            "meeting_id": s.meeting_id,
            "title": s.title,
            "capture_device": s.capture_device,
            "device_id": s.device_id,
            "start_time": s.started_at,
            "utterances": s.utterances,
        })
        if s.bookmarked:
            self.store.db.execute(
                "UPDATE memories SET importance=1.0, "
                "tags=json_insert(tags,'$[#]','bookmarked') WHERE memory_id=?",
                (memory_id,))
            self.store.db.execute(
                "UPDATE chunks SET importance=1.0 WHERE memory_id=?", (memory_id,))
            self.store.commit()
        return memory_id
