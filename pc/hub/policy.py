"""Policy Engine — privacy tags + cloud eligibility (§4 CORE / plan §2).

The one gate data must pass before leaving the device. Cloud (Sarvam ASR,
AI Cloud 100 batch) is OFF unless the user explicitly opts in AND the content's
privacy tag allows it. Cloud-bound work queues in an outbox so it is auditable
and replayable; nothing is sent implicitly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

PRIVACY_TAGS = ("normal", "sensitive", "private")


@dataclass
class PolicyEngine:
    cloud_optin: bool = False           # user-visible toggle, default OFF
    default_privacy_tag: str = "normal"
    outbox: list[dict] = field(default_factory=list)

    def is_cloud_allowed(self, privacy_tag: str | None = None) -> bool:
        tag = privacy_tag or self.default_privacy_tag
        return self.cloud_optin and tag == "normal"

    def queue_cloud_job(self, kind: str, payload: dict,
                        privacy_tag: str | None = None) -> bool:
        """Queue a batch job for the cloud tier; refused unless policy allows."""
        if not self.is_cloud_allowed(privacy_tag):
            return False
        self.outbox.append({"kind": kind, "payload": payload,
                            "queued_at": time.time(), "status": "pending"})
        return True

    def snapshot(self) -> dict:
        return {"cloud_optin": self.cloud_optin,
                "default_privacy_tag": self.default_privacy_tag,
                "outbox_pending": sum(1 for j in self.outbox
                                      if j["status"] == "pending")}

    def update(self, cloud_optin: bool | None = None,
               default_privacy_tag: str | None = None) -> dict:
        if cloud_optin is not None:
            self.cloud_optin = bool(cloud_optin)
        if default_privacy_tag is not None:
            if default_privacy_tag not in PRIVACY_TAGS:
                raise ValueError(f"privacy tag must be one of {PRIVACY_TAGS}")
            self.default_privacy_tag = default_privacy_tag
        return self.snapshot()
