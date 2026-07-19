"""TTS (text-to-speech) providers — voicing Recall's answers.

  SarvamTTSProvider   Bulbul v3 via Sarvam's REST API, policy-gated exactly
                      like Sarvam STT (never called unless PolicyEngine.
                      is_cloud_allowed() says so — this sends the ANSWER
                      text to a cloud API, the same privacy boundary as
                      sending audio).

REST contract (docs.sarvam.ai/api-reference/text-to-speech/convert):
  POST {endpoint}, header api-subscription-key, JSON body
  {text, target_language_code, model, speaker, speech_sample_rate,
   output_audio_codec, pace}. Response JSON: {"audios": ["<base64 audio>"]}.

Streaming contract (docs.sarvam.ai/api-reference/text-to-speech/stream, wire
format confirmed directly against the real endpoint — the AsyncAPI docs give
schemas but no worked JSON examples):
  wss://api.sarvam.ai/text-to-speech/ws, header api-subscription-key.
  Client sends {"type": "config"|"text"|"flush", "data": {...}}; server sends
  {"type": "audio", "data": {"request_id", "content_type", "audio": "<base64>"}}
  or {"type": "error", "data": {"message", "code", ...}}. The server never
  proactively signals "no more audio coming" for a short utterance — treated
  as done after _STREAM_IDLE_S with no new message.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import threading

from recall_memory.tracing import step

log = logging.getLogger("recall.tts")

# How long to wait for the next audio message before deciding the stream is
# over. Deliberately shorter than the REST timeout: this is the tail latency
# of every streamed answer once the LLM stops emitting text, so it directly
# trades off against the whole point of streaming.
_STREAM_IDLE_S = 6.0
# Buffer text deltas up to this many chars (or a sentence end) before
# flushing to Sarvam — flushing on every LLM token would ask it to
# synthesize word fragments one at a time instead of coherent phrases.
_MIN_FLUSH_CHARS = 40
_SENTENCE_END = ".!?\n"


class SarvamTTSProvider:
    name = "sarvam_tts"

    def __init__(self, api_key: str = "", endpoint: str = "https://api.sarvam.ai/text-to-speech",
                model: str = "bulbul:v3", speaker: str = "anushka",
                language_code: str = "en-IN", sample_rate: int = 24000,
                codec: str = "wav", pace: float = 1.0, timeout: float = 15.0,
                stream_endpoint: str = "wss://api.sarvam.ai/text-to-speech/ws",
                stream_speaker: str = "anushka"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.codec = codec
        self.pace = pace
        self.timeout = timeout
        self.stream_endpoint = stream_endpoint
        # Confirmed directly against the real endpoint (2026-07-19): the
        # streaming websocket serves bulbul:v2 regardless of the "model"
        # sent in the config message, so a v3-only speaker (e.g. the REST
        # default "shubh") gets rejected with a 422. Streaming needs its own,
        # v2-compatible speaker until Sarvam adds v3 there.
        self.stream_speaker = stream_speaker

    def synthesize(self, text: str, *, language_code: str | None = None,
                  speaker: str | None = None) -> bytes:
        """Text -> raw audio bytes (decoded from Sarvam's base64 response)."""
        if not self.api_key:
            raise RuntimeError("sarvam_tts has no RECALL_SARVAM_API_KEY configured")
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        import requests
        body = {
            "text": text[:2500],   # bulbul:v3 hard limit
            "target_language_code": language_code or self.language_code,
            "model": self.model,
            "speaker": speaker or self.speaker,
            "speech_sample_rate": self.sample_rate,
            "output_audio_codec": self.codec,
            "pace": self.pace,
        }
        with step("tts:sarvam_http_call", model=self.model,
                  chars=len(text)) as s:
            r = requests.post(
                self.endpoint,
                headers={"api-subscription-key": self.api_key,
                        "Content-Type": "application/json"},
                json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            audios = data.get("audios") or []
            s.detail(segments=len(audios))
            if not audios:
                raise RuntimeError("sarvam_tts returned no audio segments")
            # Multiple segments would need format-aware concatenation
            # (each has its own container header); bulbul:v3's 2500-char cap
            # keeps a single request to one segment in practice.
            if len(audios) > 1:
                log.warning("sarvam_tts returned %d segments; using only the "
                           "first (long-text concatenation not implemented)",
                           len(audios))
            return base64.b64decode(audios[0])

    def stream_synthesize(self, text_chunks, *, language_code: str | None = None):
        """Feed an iterable of text deltas (an LLM's streamed output, most
        likely) into Sarvam's streaming-TTS websocket as they arrive, and
        yield raw audio bytes as soon as Sarvam produces them — lets
        playback start before the full answer has even finished generating.
        A background thread drains the socket concurrently with the send
        loop below, so audio for the first sentence can arrive while later
        sentences are still being sent."""
        if not self.api_key:
            raise RuntimeError("sarvam_tts has no RECALL_SARVAM_API_KEY configured")
        from websockets.sync.client import connect

        audio_q: queue.Queue = queue.Queue()
        _done = object()

        with connect(self.stream_endpoint,
                    additional_headers={"api-subscription-key": self.api_key}) as ws:
            ws.send(json.dumps({"type": "config", "data": {
                "target_language_code": language_code or self.language_code,
                "speaker": self.stream_speaker,
                "output_audio_codec": "mp3"}}))

            def _reader():
                try:
                    while True:
                        raw = ws.recv(timeout=_STREAM_IDLE_S)
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        if mtype == "audio":
                            b64 = msg.get("data", {}).get("audio", "")
                            if b64:
                                audio_q.put(base64.b64decode(b64))
                        elif mtype == "error":
                            log.warning("sarvam tts stream error: %s", msg.get("data"))
                except Exception:
                    pass   # timeout or socket closed — treated as end of stream
                finally:
                    audio_q.put(_done)

            threading.Thread(target=_reader, daemon=True).start()

            with step("tts:sarvam_stream_send", model=self.model) as s:
                buf, buf_len, n_sent = [], 0, 0
                for chunk in text_chunks:
                    if not chunk:
                        continue
                    buf.append(chunk)
                    buf_len += len(chunk)
                    if buf_len >= _MIN_FLUSH_CHARS or chunk.strip()[-1:] in _SENTENCE_END:
                        ws.send(json.dumps({"type": "text", "data": {"text": "".join(buf)}}))
                        ws.send(json.dumps({"type": "flush"}))
                        buf, buf_len = [], 0
                        n_sent += 1
                if buf:
                    ws.send(json.dumps({"type": "text", "data": {"text": "".join(buf)}}))
                    ws.send(json.dumps({"type": "flush"}))
                    n_sent += 1
                s.detail(text_flushes=n_sent)

            while True:
                item = audio_q.get(timeout=_STREAM_IDLE_S + 2.0)
                if item is _done:
                    break
                yield item
