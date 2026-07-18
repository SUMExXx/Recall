# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0
#
# Recall — Arduino UNO Q, Linux (Dragonwing QRB2210) side.
# Implements the "Linux Side" of Architecture doc section 2:
#   ALSA capture -> 5s pre-roll ring buffer -> VAD -> wake word ("Hey Recall")
#   -> event publisher (vad / wake / button topics) + Opus/PCM audio forwarder
#   over a WebSocket to the PC hub, with a standalone buffer that flushes on
#   reconnect, and an in-process watchdog that keeps the capture service alive.
#
# The original wake-word -> LED heart flourish path (Bridge.call("keyword_detected"))
# is fully preserved; everything else is additive and degrades gracefully when a
# dependency (websocket-client, opuslib) or the PC hub is unavailable.
import subprocess
import struct
import threading
import time
import json
import math
import base64
import random
from collections import deque

import requests
from arduino.app_utils import *

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://172.21.0.2:1337/api/features"   # Edge Impulse wake classifier
DEVICE = "hw:Loopback,1,0"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000            # 1 s classification window
BYTES_PER_SAMPLE = 2
CONFIDENCE_THRESHOLD = 0.8
DEBOUNCE_SEC = 2.0

# PC hub (WebSocket). vad/wake/button events + forwarded audio are published here.
HUB_WS_URL = "ws://172.21.0.2:8765"
PRE_ROLL_SEC = 5                  # ring buffer so the wake word never clips speech
VAD_RMS_THRESHOLD = 500           # int16 RMS gate; tune to your mic/room
VAD_HANGOVER_SEC = 1.5            # keep forwarding this long after speech stops
FORWARD_MAX_SEC = 30              # hard cap on one wake-triggered forward session
OUTBOX_MAX = 1024                 # standalone buffer depth (messages)
MCU_PING_SEC = 3.0                # heartbeat to the MCU watchdog
OPUS_FRAME_SAMPLES = 320          # 20 ms @ 16 kHz

# LED states — MUST stay in sync with sketch.ino.
LED_IDLE, LED_CAPTURING, LED_SEARCHING, LED_RECALLED, LED_MUTED, LED_ERROR = 0, 1, 2, 3, 4, 5
_LED_BY_NAME = {
    "idle": LED_IDLE, "capturing": LED_CAPTURING, "searching": LED_SEARCHING,
    "recalled": LED_RECALLED, "muted": LED_MUTED, "error": LED_ERROR,
}

# Optional deps — absence must never break the wake path.
try:
    import websocket  # websocket-client
except Exception:
    websocket = None
try:
    import opuslib
except Exception:
    opuslib = None


# ---------------------------------------------------------------------------
# Hub client: event publisher + audio forwarder + standalone buffer
# ---------------------------------------------------------------------------
class HubClient:
    """WebSocket client to the PC hub. Publishes JSON events and forwarded
    audio; buffers everything in a bounded outbox while disconnected and
    flushes on reconnect (exponential backoff + jitter). Incoming LED state
    topics are dispatched to `on_led`."""

    def __init__(self, url, on_led=None):
        self.url = url
        self.on_led = on_led
        self.ws = None
        self.connected = False
        self.outbox = deque(maxlen=OUTBOX_MAX)
        self._lock = threading.Lock()
        self._opus = None
        if opuslib is not None:
            try:
                self._opus = opuslib.Encoder(SAMPLE_RATE, 1, "voip")
            except Exception as e:
                print(f"[hub] opus encoder unavailable: {e}", flush=True)
        threading.Thread(target=self._run, daemon=True).start()

    # --- connection loop -------------------------------------------------
    def _run(self):
        backoff = 1.0
        while True:
            if websocket is None:
                # No WS library: keep buffering so nothing is lost, retry later.
                time.sleep(5)
                continue
            try:
                self.ws = websocket.create_connection(self.url, timeout=5)
                self.ws.settimeout(1.0)
                self.connected = True
                backoff = 1.0
                print(f"[hub] connected {self.url}", flush=True)
                self._flush_outbox()
                self._reader_loop()
            except Exception as e:
                print(f"[hub] connect failed: {e}", flush=True)
            finally:
                self.connected = False
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.ws = None
            time.sleep(backoff + random.uniform(0, 0.5))   # jitter
            backoff = min(backoff * 2, 30)

    def _reader_loop(self):
        while self.connected:
            try:
                msg = self.ws.recv()
            except Exception as e:
                name = e.__class__.__name__
                if name == "WebSocketTimeoutException" or "timed out" in str(e).lower():
                    continue          # idle, still connected
                raise                 # real disconnect -> reconnect
            if msg:
                self._on_message(msg)

    def _on_message(self, msg):
        try:
            data = json.loads(msg)
        except Exception:
            return
        if data.get("topic") == "led" and self.on_led:
            state = data.get("state")
            if isinstance(state, str):
                state = _LED_BY_NAME.get(state.lower())
            if isinstance(state, int):
                self.on_led(state)

    # --- outgoing --------------------------------------------------------
    def publish(self, topic, payload):
        self._send(json.dumps({"topic": topic, **payload}))

    def _send(self, text):
        with self._lock:
            if self.connected and self.ws:
                try:
                    self.ws.send(text)
                    return
                except Exception:
                    self.connected = False
            self.outbox.append(text)   # buffer for flush on reconnect

    def _flush_outbox(self):
        with self._lock:
            pending = list(self.outbox)
            self.outbox.clear()
        for i, text in enumerate(pending):
            try:
                self.ws.send(text)
            except Exception:
                # Re-buffer the unsent remainder, preserving order.
                with self._lock:
                    for leftover in reversed(pending[i:]):
                        self.outbox.appendleft(leftover)
                return
        if pending:
            print(f"[hub] flushed {len(pending)} buffered messages", flush=True)

    def send_audio(self, pcm_bytes, seq):
        """Forward one PCM window. Opus if available (framed at 20 ms), else
        raw PCM (base64)."""
        if self._opus is not None:
            try:
                frame_bytes = OPUS_FRAME_SAMPLES * BYTES_PER_SAMPLE
                frames = []
                for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
                    pkt = self._opus.encode(pcm_bytes[i:i + frame_bytes], OPUS_FRAME_SAMPLES)
                    frames.append(base64.b64encode(pkt).decode())
                self.publish("audio", {"seq": seq, "codec": "opus",
                                       "rate": SAMPLE_RATE, "frames": frames})
                return
            except Exception as e:
                print(f"[hub] opus encode failed, sending PCM: {e}", flush=True)
        self.publish("audio", {"seq": seq, "codec": "pcm_s16le", "rate": SAMPLE_RATE,
                               "data": base64.b64encode(pcm_bytes).decode()})


# ---------------------------------------------------------------------------
# Capture pipeline: ring buffer -> VAD -> wake -> forward
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, hub):
        self.hub = hub
        self.pre_roll = deque(maxlen=PRE_ROLL_SEC)   # last N 1-s PCM windows
        self.last_trigger = 0.0
        self.speaking = False
        self.session_active = False
        self.forward_until = 0.0
        self.session_end = 0.0
        self.seq = 0
        self.muted = False
        self._led = LED_IDLE

    # Hardware mute (from the MCU switch): gate everything leaving the device.
    def set_muted(self, muted):
        self.muted = bool(muted)
        if self.muted:
            self.session_active = False
        self.set_led(LED_MUTED if self.muted else LED_IDLE)

    def set_led(self, state):
        if state == self._led:
            return
        self._led = state
        try:
            Bridge.call("set_led_state", state)
        except Exception as e:
            print(f"[led] {e}", flush=True)

    @staticmethod
    def _rms(samples):
        if not samples:
            return 0.0
        acc = 0
        for s in samples:
            acc += s * s
        return math.sqrt(acc / len(samples))

    def _classify(self, samples):
        """Existing Edge Impulse wake-word classifier. Returns True on 'Hey Recall'."""
        try:
            resp = requests.post(API_URL, json={"features": list(samples)}, timeout=5)
            classification = resp.json().get("result", {}).get("classification", {})
            if classification:
                top_label = max(classification, key=classification.get)
                top_score = classification[top_label]
                print(f"{classification} --> top: {top_label} ({top_score:.2f})", flush=True)
                return top_label == "hey_arduino" and top_score > CONFIDENCE_THRESHOLD
        except Exception as e:
            print(f"Request failed: {e}", flush=True)
        return False

    def process_window(self, window, samples):
        now = time.time()
        self.pre_roll.append(window)

        if self.muted:
            return   # physical privacy: no VAD, no wake, no audio leaves

        # ---- VAD gate ----
        is_speech = self._rms(samples) >= VAD_RMS_THRESHOLD
        if is_speech and not self.speaking:
            self.speaking = True
            self.hub.publish("vad", {"event": "speech_start", "ts": now})
        elif not is_speech and self.speaking:
            self.speaking = False
            self.hub.publish("vad", {"event": "speech_end", "ts": now})

        # ---- wake word ----
        if self._classify(samples) and (now - self.last_trigger) > DEBOUNCE_SEC:
            self.last_trigger = now
            self._on_wake(now)

        # ---- audio forwarding (only during a wake-triggered session) ----
        if self.session_active:
            if is_speech:
                self.forward_until = now + VAD_HANGOVER_SEC
            self._forward(window)
            if now >= self.forward_until or now >= self.session_end:
                self.session_active = False
                self.hub.publish("wake", {"event": "capture_end", "ts": now})
                self.set_led(LED_IDLE)

    def _on_wake(self, now):
        print("[wake] Hey Recall", flush=True)
        try:
            Bridge.call("keyword_detected")          # preserved heart flourish
        except Exception as e:
            print(f"[bridge] {e}", flush=True)
        self.set_led(LED_CAPTURING)
        self.hub.publish("wake", {"event": "detected", "phrase": "hey_recall", "ts": now})
        # Open a forward session and flush the pre-roll first so the utterance
        # that triggered the wake word is never clipped.
        self.session_active = True
        self.forward_until = now + VAD_HANGOVER_SEC
        self.session_end = now + FORWARD_MAX_SEC
        for w in list(self.pre_roll):
            self._forward(w)

    def _forward(self, window):
        self.seq += 1
        self.hub.send_audio(window, self.seq)


# ---------------------------------------------------------------------------
# ALSA capture (supervised)
# ---------------------------------------------------------------------------
def capture_and_classify(pipeline):
    cmd = ["arecord", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
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
    """tap -> 'bookmark', hold 3 s -> 'forget'. Published to the hub."""
    action = str(action)
    print(f"[button] {action}", flush=True)
    if action == "forget":
        hub.publish("button", {"action": "forget_range", "seconds": 300, "ts": time.time()})
    else:
        hub.publish("button", {"action": "bookmark", "ts": time.time()})


def on_mute_changed(muted):
    print(f"[mute] {'muted' if muted else 'unmuted'}", flush=True)
    pipeline.set_muted(muted)
    hub.publish("mute", {"muted": bool(muted), "ts": time.time()})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
hub = HubClient(HUB_WS_URL, on_led=lambda s: pipeline.set_led(s))
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
