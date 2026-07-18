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
#                   button{action: bookmark | forget_last, minutes?}
#   hub -> client : welcome{resume_token, seq, missed[], led_state} | pong
#                   event{topic: led_state|transcript|recall|answer, seq, data}
#
# Flow: ALSA capture -> 5 s pre-roll ring buffer -> local VAD + "Hey Recall"
# wake word. On wake we open a meeting (LED -> capturing on the hub), stream the
# pre-roll + live audio as raw PCM16 binary frames, and close the meeting after a
# silence hangover so the hub ingests it as one memory. LED state is driven by
# the hub's led_state events, forwarded to the MCU over the RPC bridge. The
# hardware mute switch gates audio locally (physical privacy).
#
# The original wake-word -> LED heart flourish (Bridge.call("keyword_detected"))
# is preserved. Everything degrades gracefully if websocket-client or the hub is
# unavailable — the wake -> heart path keeps working offline.
import json
import math
import os
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
HUB_WS_URL = os.environ.get("RECALL_HUB_WS", "ws://127.0.0.1:8000/ws")
DEVICE_ID = os.environ.get("RECALL_DEVICE_ID", "unoq-01")
ROLE = "unoq"                     # pc/hub/registry.py roles: phone|unoq|dashboard|mcp|other

PRE_ROLL_SEC = 5                  # ring buffer so the wake word never clips speech
VAD_RMS_THRESHOLD = 500           # int16 RMS gate; tune to your mic/room
VAD_HANGOVER_SEC = 1.5            # end the meeting this long after speech stops
MEETING_MAX_SEC = 120             # hard cap on a single wake-triggered meeting
HEARTBEAT_SEC = 10.0             # < registry LEASE_SECONDS (30) in pc/hub/registry.py
MCU_PING_SEC = 3.0                # heartbeat to the MCU watchdog
OUTBOX_MAX = 256                  # standalone buffer for control messages

# LED states — indices MUST match sketch.ino; names MUST match hub LED_STATES.
LED_STATES = ("idle", "capturing", "searching", "recalled", "muted", "error")
LED_INDEX = {name: i for i, name in enumerate(LED_STATES)}

try:
    import websocket  # websocket-client
except Exception:
    websocket = None


# ---------------------------------------------------------------------------
# Hub client — real pc/hub protocol
# ---------------------------------------------------------------------------
class HubClient:
    """WebSocket client to the PC hub. Handles the hello/welcome handshake,
    heartbeats, meeting lifecycle, binary audio streaming and led_state events.
    Control messages are buffered in a bounded outbox while disconnected and
    flushed on reconnect (exponential backoff + jitter)."""

    def __init__(self, url, on_led=None):
        self.url = url
        self.on_led = on_led
        self.ws = None
        self.connected = False
        self.resume_token = None
        self.last_seq = 0
        self.in_meeting = False
        self.outbox = deque(maxlen=OUTBOX_MAX)
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

    # --- connection ------------------------------------------------------
    def _run(self):
        backoff = 1.0
        while True:
            if websocket is None:
                time.sleep(5)
                continue
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
                self.connected = False
                self.in_meeting = False
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
            time.sleep(backoff + (time.time() % 0.5))   # small jitter
            backoff = min(backoff * 2, 30)

    def _handshake(self):
        hello = {"type": "hello", "device_id": DEVICE_ID, "role": ROLE}
        if self.resume_token:
            hello["resume_token"] = self.resume_token
            hello["last_seq"] = self.last_seq
        self.ws.send(json.dumps(hello))
        self.ws.settimeout(5)
        welcome = json.loads(self.ws.recv())
        if welcome.get("type") == "welcome":
            self.resume_token = welcome.get("resume_token", self.resume_token)
            self.last_seq = welcome.get("seq", self.last_seq)
            for evt in welcome.get("missed", []):
                self._dispatch_event(evt)
            if welcome.get("led_state"):
                self._emit_led(welcome["led_state"])
        # Only care about LED state topics from the hub.
        self.ws.send(json.dumps({"type": "subscribe", "topics": ["led_state"]}))
        self.ws.settimeout(1.0)

    def _reader_loop(self):
        while self.connected:
            try:
                raw = self.ws.recv()
            except Exception as e:
                name = e.__class__.__name__
                if name == "WebSocketTimeoutException" or "timed out" in str(e).lower():
                    continue
                raise
            if not raw:
                continue
            if isinstance(raw, bytes):
                continue   # the hub never sends us binary
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
            self.in_meeting = False
        # pong / ack / meeting_started are informational.

    def _dispatch_event(self, evt):
        if evt.get("seq"):
            self.last_seq = max(self.last_seq, evt["seq"])
        if evt.get("topic") == "led_state":
            self._emit_led(evt.get("data", {}).get("state"))

    def _emit_led(self, state):
        if isinstance(state, str) and state in LED_INDEX and self.on_led:
            self.on_led(LED_INDEX[state])

    # --- outgoing --------------------------------------------------------
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
                self.outbox.append(text)
        return False

    def _flush_outbox(self):
        with self._lock:
            pending = list(self.outbox)
            self.outbox.clear()
        for i, text in enumerate(pending):
            try:
                self.ws.send(text)
            except Exception:
                with self._lock:
                    for leftover in reversed(pending[i:]):
                        self.outbox.appendleft(leftover)
                return
        if pending:
            print(f"[hub] flushed {len(pending)} buffered messages", flush=True)

    def _heartbeat_loop(self):
        while True:
            time.sleep(HEARTBEAT_SEC)
            if self.connected:
                self._send_json({"type": "heartbeat"}, buffer_if_down=False)

    # --- meeting lifecycle ----------------------------------------------
    def meeting_start(self):
        if self.in_meeting:
            return
        self.in_meeting = True
        self._send_json({"type": "meeting_start", "capture_device": "arduino"},
                        buffer_if_down=False)

    def send_audio(self, pcm_bytes):
        if not (self.connected and self.ws and self.in_meeting):
            return
        with self._lock:
            try:
                self.ws.send_binary(pcm_bytes)   # raw PCM16, per hub protocol
            except Exception:
                self.connected = False

    def meeting_end(self):
        if not self.in_meeting:
            return
        self.in_meeting = False
        self._send_json({"type": "meeting_end"}, buffer_if_down=False)

    def button(self, action, minutes=None):
        msg = {"type": "button", "action": action}
        if minutes is not None:
            msg["minutes"] = minutes
        self._send_json(msg)   # buffered if offline — user intent is worth keeping


# ---------------------------------------------------------------------------
# Capture pipeline: ring buffer -> VAD -> wake -> stream meeting
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, hub):
        self.hub = hub
        self.pre_roll = deque(maxlen=PRE_ROLL_SEC)   # last N 1-s PCM windows
        self.last_trigger = 0.0
        self.forward_until = 0.0
        self.meeting_deadline = 0.0
        self.muted = False
        self._led = LED_INDEX["idle"]

    def set_muted(self, muted):
        """Hardware mute switch: gate all audio locally (physical privacy)."""
        self.muted = bool(muted)
        if self.muted and self.hub.in_meeting:
            self.hub.meeting_end()
        self.set_led(LED_INDEX["muted"] if self.muted else LED_INDEX["idle"])

    def set_led(self, index):
        if index == self._led:
            return
        self._led = index
        try:
            Bridge.call("set_led_state", index)
        except Exception as e:
            print(f"[led] {e}", flush=True)

    @staticmethod
    def _rms(samples):
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    def _is_wake(self, samples):
        try:
            resp = requests.post(API_URL, json={"features": list(samples)}, timeout=5)
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
        self.pre_roll.append(window)

        if self.muted:
            return   # physical privacy: nothing leaves the device

        is_speech = self._rms(samples) >= VAD_RMS_THRESHOLD

        # ---- wake word opens a meeting ----
        if self._is_wake(samples) and (now - self.last_trigger) > DEBOUNCE_SEC:
            self.last_trigger = now
            self._on_wake(now)

        # ---- stream + close the meeting on a silence hangover ----
        if self.hub.in_meeting:
            if is_speech:
                self.forward_until = now + VAD_HANGOVER_SEC
            self.hub.send_audio(window)
            if now >= self.forward_until or now >= self.meeting_deadline:
                self.hub.meeting_end()
                self.set_led(LED_INDEX["idle"])

    def _on_wake(self, now):
        print("[wake] Hey Recall", flush=True)
        try:
            Bridge.call("keyword_detected")          # preserved heart flourish
        except Exception as e:
            print(f"[bridge] {e}", flush=True)
        self.set_led(LED_INDEX["capturing"])         # hub will confirm via led_state
        self.hub.meeting_start()
        self.forward_until = now + VAD_HANGOVER_SEC
        self.meeting_deadline = now + MEETING_MAX_SEC
        for w in list(self.pre_roll):                # never clip the triggering speech
            self.hub.send_audio(w)


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
    while True:
        try:
            capture_and_classify(pipeline)
        except Exception as e:
            print(f"[supervisor] capture crashed: {e}", flush=True)
        print("[supervisor] restarting capture in 2s", flush=True)
        time.sleep(2)


# ---------------------------------------------------------------------------
# MCU button / mute events (provided to the sketch via the RPC bridge)
# ---------------------------------------------------------------------------
def on_button(action):
    """tap -> 'bookmark', hold 3 s -> 'forget'. Mapped to hub button actions."""
    action = str(action)
    print(f"[button] {action}", flush=True)
    if action == "forget":
        hub.button("forget_last", minutes=5)
    else:
        hub.button("bookmark")


def on_mute_changed(muted):
    print(f"[mute] {'muted' if muted else 'unmuted'}", flush=True)
    pipeline.set_muted(muted)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
hub = HubClient(HUB_WS_URL, on_led=lambda i: pipeline.set_led(i))
pipeline = Pipeline(hub)

# The sketch calls these over the bridge (buttons + hardware mute switch).
Bridge.provide("button_event", on_button)
Bridge.provide("mute_changed", on_mute_changed)

# Supervised ALSA capture in the background so App.run() owns the main thread.
threading.Thread(target=capture_supervisor, args=(pipeline,), daemon=True).start()

_last_ping = 0.0


def loop():
    """Heartbeat so the MCU watchdog knows the Linux side is alive."""
    global _last_ping
    now = time.time()
    if now - _last_ping >= MCU_PING_SEC:
        _last_ping = now
        try:
            Bridge.call("mcu_ping")
        except Exception:
            pass


App.run(user_loop=loop)
