# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
#
# Recall — Arduino UNO Q, Linux (Dragonwing QRB2210) side.
#
# Speaks the PC Hub WebSocket protocol defined in pc/hub/app.py:
#   client -> hub : hello{device_id, role, resume_token?, last_seq?}
#                   heartbeat{} | subscribe{topics[]}
#                   meeting_start{capture_device} -> <binary PCM16 frames> -> meeting_end{}
#   hub -> client : welcome{resume_token, seq, missed[], led_state} | pong
#                   event{topic: led_state|transcript|recall|answer, seq, data}
#
# (Push-button bookmark/forget and the hardware mute switch are intentionally
#  not implemented.)
#
# Flow: ALSA capture -> "Hey Recall" wake word. On wake we open a meeting
# (LED -> capturing on the hub), stream live audio as raw PCM16 binary frames,
# and close the meeting after a fixed capture window so the hub ingests it as one
# memory. LED policy: this device OWNS idle/capturing (driven locally by its own
# capture state), while the hub drives searching/recalled/error/muted as transient
# overrides. The hub's own idle/capturing are ignored, so a late hub transcription
# (which can auto-start a session on the hub) can't spuriously show "capturing".
#
# The original wake-word -> LED heart flourish (Bridge.call("keyword_detected"))
# is preserved. Everything degrades gracefully if websocket-client or the hub is
# unavailable — the wake -> heart path keeps working offline.
import json
import os
import random
import struct
import subprocess
import threading
import time
from collections import deque

import requests
from arduino.app_utils import *

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://172.21.0.2:1337/api/features"   # on-board Edge Impulse wake classifier
DEVICE = "hw:Loopback,1,0"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000            # 1 s classification / stream window
BYTES_PER_SAMPLE = 2
CONFIDENCE_THRESHOLD = 0.8
DEBOUNCE_SEC = 2.0

# PC Hub WebSocket. The hub binds RECALL_HOST:RECALL_PORT (default 127.0.0.1:8000
# in pc/hub/__main__.py); for a separate UNO Q on the LAN, point this at the hub
# PC's address and run the hub with RECALL_HOST=0.0.0.0. Override via RECALL_HUB_WS.
HUB_WS_URL = os.environ.get("RECALL_HUB_WS", "ws://10.92.169.190:8000/ws")
DEVICE_ID = os.environ.get("RECALL_DEVICE_ID", "unoq-01")
ROLE = "unoq"                     # pc/hub/registry.py roles: phone|unoq|dashboard|mcp|other

CAPTURE_SEC = 10                  # FIXED capture length after wake — never extended by speech
HEARTBEAT_SEC = 5.0               # << registry LEASE_SECONDS (30); survives a stalled beat
HANDSHAKE_TIMEOUT = 5.0           # recv timeout while waiting for `welcome`
MCU_PING_SEC = 3.0                # heartbeat to the MCU watchdog
OUTBOX_MAX_SEC = 10.0             # buffer control messages for up to N s of disconnection
                                  # (anything older is stale on reconnect and dropped)
OUTBOX_HARD_MAX = 1024            # count backstop so a flood can't grow memory unbounded

# LED states — indices MUST match sketch.ino; names MUST match hub LED_STATES.
LED_STATES = ("idle", "capturing", "searching", "recalled", "muted", "error")
LED_INDEX = {name: i for i, name in enumerate(LED_STATES)}
# States the HUB owns (transient overrides we display verbatim). idle/capturing
# are owned LOCALLY by this device, so the hub's opinion on those is ignored —
# that's what stops a late hub transcription from spuriously showing "capturing".
HUB_OVERRIDE_STATES = ("searching", "recalled", "muted", "error")

try:
    import websocket  # websocket-client
    from websocket import WebSocketTimeoutException
except Exception:
    websocket = None

    class WebSocketTimeoutException(Exception):   # fallback so `except` stays valid
        pass


# ---------------------------------------------------------------------------
# Hub client — real pc/hub protocol
# ---------------------------------------------------------------------------
class HubClient:
    """WebSocket client to the PC hub. Handles the hello/welcome handshake,
    heartbeats, meeting lifecycle, binary audio streaming and led_state events
    (forwarded to `on_hub_state`; the Pipeline decides how to apply them).
    Control messages are buffered in a bounded outbox while disconnected and
    flushed on reconnect (exponential backoff + jitter).

    Threading: `_run` (reader) and `_heartbeat_loop` run on their own daemon
    threads; the capture thread calls meeting_start/send_audio/meeting_end. A
    single reentrant lock (`_lock`) guards both the `in_meeting` flag and all
    socket writes, so those cross-thread accesses are serialized (bugs #2, #5)."""

    def __init__(self, url, on_hub_state=None):
        self.url = url
        self.on_hub_state = on_hub_state   # called with each hub led_state name
        self.ws = None
        self.connected = False
        self.resume_token = None
        self.last_seq = 0
        self.in_meeting = False
        self.outbox = deque(maxlen=OUTBOX_HARD_MAX)   # holds (enqueued_at, text)
        self._lock = threading.RLock()   # reentrant: send paths may nest (bug #5)
        threading.Thread(target=self._run, daemon=True,
                         name="hub-reader").start()
        threading.Thread(target=self._heartbeat_loop, daemon=True,
                         name="hub-heartbeat").start()

    # --- connection ------------------------------------------------------
    def _run(self):
        if websocket is None:
            print("[hub] websocket-client not installed — hub streaming disabled; "
                  "wake -> heart still works", flush=True)
            return
        backoff = 1.0
        while True:
            try:
                self.ws = websocket.create_connection(self.url, timeout=5)
                self._handshake()
                self.connected = True
                backoff = 1.0
                print(f"[hub] connected {self.url}", flush=True)
                self._flush_outbox()
                self._reader_loop()
            except Exception as e:
                print(f"[hub] connection error: {e}", flush=True)
            finally:
                with self._lock:
                    self.connected = False
                    self.in_meeting = False
                    try:
                        if self.ws:
                            self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
            time.sleep(backoff + random.uniform(0, 0.5))   # jitter (bug #17)
            backoff = min(backoff * 2, 30)

    def _handshake(self):
        hello = {"type": "hello", "device_id": DEVICE_ID, "role": ROLE}
        if self.resume_token:
            hello["resume_token"] = self.resume_token
            hello["last_seq"] = self.last_seq
        self.ws.settimeout(HANDSHAKE_TIMEOUT)   # bound the wait for `welcome` (bug #9)
        self.ws.send(json.dumps(hello))
        welcome = json.loads(self.ws.recv())
        if welcome.get("type") == "welcome":
            self.resume_token = welcome.get("resume_token", self.resume_token)
            self.last_seq = welcome.get("seq", self.last_seq)
            for evt in welcome.get("missed", []):
                self._dispatch_event(evt)
            if welcome.get("led_state") and self.on_hub_state:
                self.on_hub_state(welcome["led_state"])
        # Only the led_state topic — we display the hub's searching/recalled/error.
        self.ws.send(json.dumps({"type": "subscribe", "topics": ["led_state"]}))
        self.ws.settimeout(1.0)                 # short recv timeout for the reader loop

    def _reader_loop(self):
        while self.connected:
            try:
                raw = self.ws.recv()
            except WebSocketTimeoutException:
                continue                        # idle tick, still connected (bug #10)
            if not raw:
                continue
            if isinstance(raw, bytes):
                continue                        # the hub never sends us binary
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            self._on_message(msg)

    def _on_message(self, msg):
        mtype = msg.get("type")
        if mtype == "event":
            self._dispatch_event(msg)
        elif mtype == "meeting_ended":
            with self._lock:
                self.in_meeting = False
        # pong / ack / meeting_started are informational.

    def _dispatch_event(self, evt):
        if evt.get("seq"):
            self.last_seq = max(self.last_seq, evt["seq"])   # keep resume cursor current
        if evt.get("topic") == "led_state" and self.on_hub_state:
            state = evt.get("data", {}).get("state")
            if isinstance(state, str):
                self.on_hub_state(state)

    # --- outgoing --------------------------------------------------------
    def _prune_outbox(self, now=None):
        """Drop buffered messages older than OUTBOX_MAX_SEC (stale on reconnect).
        Entries are (enqueued_at, text) in FIFO order, so the oldest are at the
        left. Caller must hold self._lock."""
        cutoff = (now or time.time()) - OUTBOX_MAX_SEC
        while self.outbox and self.outbox[0][0] < cutoff:
            self.outbox.popleft()

    def _send_json(self, obj, buffer_if_down=True):
        text = json.dumps(obj)
        with self._lock:
            if self.connected and self.ws:
                try:
                    self.ws.send(text)
                    return True
                except Exception:
                    self.connected = False
            if buffer_if_down:
                self.outbox.append((time.time(), text))   # timestamp for age-based pruning
                self._prune_outbox()
        return False

    def _flush_outbox(self):
        # Whole flush under the lock so the capture thread can't interleave new
        # sends or appends mid-drain, keeping strict FIFO order (bug #8).
        sent = 0
        with self._lock:
            self._prune_outbox()                # drop what went stale while disconnected
            pending = list(self.outbox)         # list of (enqueued_at, text)
            self.outbox.clear()
            for sent, (_, text) in enumerate(pending):
                try:
                    self.ws.send(text)
                except Exception:
                    # Re-queue the unsent remainder, keeping original timestamps
                    # so their age keeps counting from when they were first queued.
                    for leftover in reversed(pending[sent:]):
                        self.outbox.appendleft(leftover)
                    sent = -1
                    break
            else:
                sent = len(pending)
        if sent > 0:
            print(f"[hub] flushed {sent} buffered messages", flush=True)

    def _heartbeat_loop(self):
        while True:
            time.sleep(HEARTBEAT_SEC)
            if self.connected:
                self._send_json({"type": "heartbeat"}, buffer_if_down=False)

    # --- meeting lifecycle ----------------------------------------------
    def meeting_start(self):
        with self._lock:                        # atomic check-then-set (bug #2)
            if self.in_meeting:
                return
            self.in_meeting = True
        self._send_json({"type": "meeting_start", "capture_device": "arduino"},
                        buffer_if_down=False)

    def send_audio(self, pcm_bytes):
        with self._lock:                        # read flag + write frame atomically (bug #2)
            if not (self.connected and self.ws and self.in_meeting):
                return
            try:
                self.ws.send_binary(pcm_bytes)  # raw PCM16, per hub protocol
            except Exception:
                self.connected = False

    def meeting_end(self):
        with self._lock:
            if not self.in_meeting:
                return
            self.in_meeting = False
        self._send_json({"type": "meeting_end"}, buffer_if_down=False)


# ---------------------------------------------------------------------------
# Capture pipeline: wake word -> stream a fixed-length meeting
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, hub):
        self.hub = hub                          # may be set after construction (bug #1)
        self.last_trigger = 0.0
        self.capture_until = 0.0
        # LED controller. This device owns idle/capturing (local); the hub drives
        # searching/recalled/error as transient OVERRIDES. Touched by both the
        # capture thread (set_local_led) and the hub reader thread (on_hub_led),
        # so guard the decision with a lock.
        self._led_lock = threading.Lock()
        self._local_led = LED_INDEX["idle"]     # our own state: idle or capturing
        self._override = None                   # active hub transient index, or None
        self._shown = LED_INDEX["idle"]         # what's currently on the MCU

    def set_local_led(self, index):
        """Our own capture state (idle/capturing). Shows it unless a hub transient
        is currently overriding — in which case it takes effect once that clears."""
        with self._led_lock:
            self._local_led = index
            if self._override is None:
                self._render_locked(index)

    def on_hub_led(self, name):
        """Hub-driven led_state. searching/recalled/error/muted OVERRIDE the LED;
        the hub's own idle/capturing are ignored (this device owns those) and just
        clear the override, reverting to our local state. That is what prevents a
        late hub transcription's spurious 'capturing' from showing here."""
        with self._led_lock:
            if name in HUB_OVERRIDE_STATES:
                self._override = LED_INDEX[name]
                self._render_locked(self._override)
            elif name in ("idle", "capturing"):
                self._override = None
                self._render_locked(self._local_led)

    def _render_locked(self, index):
        if index == self._shown:
            return
        self._shown = index
        try:
            Bridge.call("set_led_state", index)
        except Exception as e:
            print(f"[led] {e}", flush=True)

    def _is_wake(self, samples):
        try:
            resp = requests.post(API_URL, json={"features": samples}, timeout=5)
            classification = resp.json().get("result", {}).get("classification", {})
            if classification:
                top = max(classification, key=classification.get)
                score = classification[top]
                print(f"{classification} --> top: {top} ({score:.2f})", flush=True)
                return top == "hey_arduino" and score > CONFIDENCE_THRESHOLD
        except Exception as e:
            print(f"Request failed: {e}", flush=True)
        return False

    def process_window(self, window, samples):
        now = time.time()

        # While a meeting is live, stream for a FIXED window then stop. Continued
        # speech does NOT extend it — no VAD hangover, no loop. (The blocking wake
        # classifier is also skipped mid-meeting so it can't stall arecord, bug #3.)
        if self.hub.in_meeting:
            self.hub.send_audio(window)
            # Re-check in_meeting so we don't double-end / redundantly set the LED
            # if the hub already closed the meeting (bug #11).
            if now >= self.capture_until and self.hub.in_meeting:
                self.hub.meeting_end()
                self.set_local_led(LED_INDEX["idle"])
                # Cooldown: don't let the tail of this utterance immediately
                # re-trigger the wake word right after the meeting closes.
                self.last_trigger = now
            return

        # Idle: listen for the wake word.
        if self._is_wake(samples) and (now - self.last_trigger) > DEBOUNCE_SEC:
            self.last_trigger = now
            self._on_wake(now)

    def _on_wake(self, now):
        print("[wake] Hey Recall", flush=True)
        try:
            Bridge.call("keyword_detected")          # preserved heart flourish
        except Exception as e:
            print(f"[bridge] {e}", flush=True)
        self.set_local_led(LED_INDEX["capturing"])   # local LED; idle is set on meeting end
        self.hub.meeting_start()
        self.capture_until = now + CAPTURE_SEC       # fixed window; not extended by speech


# ---------------------------------------------------------------------------
# ALSA capture (supervised)
# ---------------------------------------------------------------------------
def capture_and_classify(pipeline):
    cmd = ["arecord", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
           "-c", "1", "-t", "raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    bytes_needed = WINDOW_SAMPLES * BYTES_PER_SAMPLE
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(bytes_needed - len(buf))
            if not chunk:
                print("No audio data - check the loopback stream is running.", flush=True)
                break
            buf += chunk
            if len(buf) >= bytes_needed:
                window = buf[:bytes_needed]
                buf = buf[bytes_needed:]
                samples = struct.unpack(f"<{WINDOW_SAMPLES}h", window)
                pipeline.process_window(window, samples)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def capture_supervisor(pipeline):
    """systemd-watchdog equivalent, in-process: restart capture if it dies."""
    restarts = 0
    while True:
        try:
            capture_and_classify(pipeline)
        except Exception as e:
            print(f"[supervisor] capture crashed: {e}", flush=True)
        restarts += 1
        print(f"[supervisor] restarting capture in 2s (restart #{restarts})", flush=True)
        time.sleep(2)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
# Pipeline holds a reference to the hub; build the hub, then hand it to the
# pipeline. (The capture thread — started below — is the only user of both.)
pipeline = Pipeline(None)
hub = HubClient(HUB_WS_URL, on_hub_state=pipeline.on_hub_led)
pipeline.hub = hub

if websocket is None:
    print("[hub] websocket-client not installed — hub streaming disabled", flush=True)

# Supervised ALSA capture in the background so App.run() owns the main thread.
threading.Thread(target=capture_supervisor, args=(pipeline,), daemon=True,
                 name="capture-supervisor").start()

_last_ping = 0.0
_last_ping_err = 0.0


def loop():
    """Heartbeat so the MCU watchdog knows the Linux side is alive."""
    global _last_ping, _last_ping_err
    now = time.time()
    if now - _last_ping >= MCU_PING_SEC:
        _last_ping = now
        try:
            Bridge.call("mcu_ping")
        except Exception as e:
            if now - _last_ping_err > 30:        # log, but don't spam every 3 s (bug #19)
                _last_ping_err = now
                print(f"[mcu] ping failed: {e}", flush=True)


App.run(user_loop=loop)
