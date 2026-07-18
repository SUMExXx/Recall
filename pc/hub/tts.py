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
"""
from __future__ import annotations

import base64
import logging

from recall_memory.tracing import step

log = logging.getLogger("recall.tts")


class SarvamTTSProvider:
    name = "sarvam_tts"

    def __init__(self, api_key: str = "", endpoint: str = "https://api.sarvam.ai/text-to-speech",
                model: str = "bulbul:v3", speaker: str = "anushka",
                language_code: str = "en-IN", sample_rate: int = 24000,
                codec: str = "wav", pace: float = 1.0, timeout: float = 15.0):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.speaker = speaker
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.codec = codec
        self.pace = pace
        self.timeout = timeout

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
