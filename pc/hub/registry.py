"""Device Registry — reliability layer (§4 REL).

Tracks every connected device (phone, unoq, dashboard, mcp), its heartbeats,
and a lease that expires if heartbeats stop. mDNS advertisement is the event-PC
transport concern; on the dev laptop devices connect straight to the WS URL.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

LEASE_SECONDS = 30.0


@dataclass
class Device:
    device_id: str
    role: str                      # phone | unoq | dashboard | mcp | other
    resume_token: str
    connected: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    last_seq_seen: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def lease_expired(self) -> bool:
        return (time.time() - self.last_heartbeat) > LEASE_SECONDS

    def snapshot(self) -> dict:
        return {
            "device_id": self.device_id, "role": self.role,
            "connected": self.connected and not self.lease_expired,
            "last_heartbeat": round(self.last_heartbeat, 1),
            "lease_expired": self.lease_expired,
        }


class DeviceRegistry:
    def __init__(self):
        self.devices: dict[str, Device] = {}

    def hello(self, device_id: str, role: str,
              resume_token: str | None = None) -> tuple[Device, bool]:
        """Register/re-register a device. Returns (device, resumed)."""
        existing = self.devices.get(device_id)
        resumed = bool(existing and resume_token
                       and existing.resume_token == resume_token)
        if existing and resumed:
            existing.connected = True
            existing.last_heartbeat = time.time()
            return existing, True
        dev = Device(device_id=device_id, role=role,
                     resume_token=secrets.token_hex(8))
        self.devices[device_id] = dev
        return dev, False

    def heartbeat(self, device_id: str):
        dev = self.devices.get(device_id)
        if dev:
            dev.last_heartbeat = time.time()

    def disconnect(self, device_id: str):
        dev = self.devices.get(device_id)
        if dev:
            dev.connected = False

    def sweep(self) -> list[str]:
        """Mark lease-expired devices disconnected; returns their ids."""
        expired = [d.device_id for d in self.devices.values()
                   if d.connected and d.lease_expired]
        for did in expired:
            self.devices[did].connected = False
        return expired

    def snapshot(self) -> list[dict]:
        return [d.snapshot() for d in self.devices.values()]
