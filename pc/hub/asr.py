"""ASRProvider workers — plan §2's interface, hub-side.

Providers:
  WhisperLocalProvider    posts WAV bytes to a whisper.cpp-contract /inference
                          endpoint (dev: server/scripts/dev_whisper_server.py
                          from Recall 1.0; event PC: ORT-QNN Whisper-Base-En)
  PassthroughProvider     dev shortcut for clients that send text utterances
  HinglishLocalProvider   BYOM Oriserve slot (falls back to Whisper here)
  SarvamCloudProvider     policy-gated opt-in; NEVER called unless the policy
                          engine allows it (3s timeout -> Hinglish -> Whisper)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ASRResult:
    text: str
    lang: str = "en"
    segments: list = field(default_factory=list)   # [{t_start, t_end, text}]
    confidence: float = 1.0                        # unified 0..1


class PassthroughProvider:
    """Client already transcribed (or dev sends text): normalize into ASRResult."""
    name = "passthrough"

    def transcribe(self, payload: dict) -> ASRResult:
        return ASRResult(text=payload["text"], lang=payload.get("lang", "en"),
                         confidence=float(payload.get("asr_confidence", 1.0)))

    def capabilities(self) -> dict:
        return {"langs": ["*"], "word_ts": False, "diarization": False,
                "offline": True, "codemix": True}


class WhisperLocalProvider:
    name = "whisper_local"

    def __init__(self, url: str = "http://localhost:8080"):
        self.url = url.rstrip("/")

    def transcribe(self, payload: dict) -> ASRResult:
        """payload = {audio: bytes (wav), lang?}. whisper.cpp /inference contract."""
        import requests
        r = requests.post(
            f"{self.url}/inference",
            files={"file": ("audio.wav", payload["audio"], "audio/wav")},
            data={"response_format": "json"},
            timeout=120)
        r.raise_for_status()
        data = r.json()
        segs = data.get("segments", [])
        # confidence = exp(mean segment logprob), normalized (plan §2)
        logprobs = [s.get("avg_logprob") for s in segs
                    if s.get("avg_logprob") is not None]
        conf = min(1.0, math.exp(sum(logprobs) / len(logprobs))) if logprobs else 0.9
        return ASRResult(
            text=data.get("text", "").strip(),
            lang=data.get("language", payload.get("lang", "en")),
            segments=[{"t_start": s.get("start", s.get("t0", 0)),
                       "t_end": s.get("end", s.get("t1", 0)),
                       "text": s.get("text", "")} for s in segs],
            confidence=conf)

    def capabilities(self) -> dict:
        return {"langs": ["en"], "word_ts": False, "diarization": False,
                "offline": True, "codemix": False}


class HinglishLocalProvider(WhisperLocalProvider):
    """BYOM Oriserve Whisper-Hindi2Hinglish slot — same transport contract,
    different model behind the endpoint on the event PC."""
    name = "hinglish_local"

    def capabilities(self) -> dict:
        return {"langs": ["hi", "en"], "word_ts": False, "diarization": False,
                "offline": True, "codemix": True}


class SarvamCloudProvider:
    """Opt-in cloud ASR. Deliberately unimplemented on the dev laptop: the
    selection policy treats any exception as a timeout and falls back locally."""
    name = "sarvam_cloud"

    def transcribe(self, payload: dict) -> ASRResult:
        raise RuntimeError("Sarvam cloud provider not wired on dev machine")

    def capabilities(self) -> dict:
        return {"langs": ["hi", "en", "+21 indic"], "word_ts": True,
                "diarization": False, "offline": False, "codemix": True}


def select_provider(lang: str, lang_prob: float, policy, providers: dict,
                    privacy_tag: str | None = None):
    """Plan §2 selection policy. providers = {name: provider}."""
    if lang == "en" or lang_prob < 0.5:
        return providers["whisper_local"]
    indic = lang in ("hi", "hi-en")
    if indic and policy.is_cloud_allowed(privacy_tag) \
            and "sarvam_cloud" in providers:
        return providers["sarvam_cloud"]   # caller wraps with timeout+fallback
    if indic and "hinglish_local" in providers:
        return providers["hinglish_local"]
    return providers["whisper_local"]


def transcribe_with_fallback(payload: dict, lang: str, lang_prob: float,
                             policy, providers: dict,
                             privacy_tag: str | None = None) -> ASRResult:
    """Fallback chain: selected -> hinglish_local -> whisper_local -> passthrough."""
    order = [select_provider(lang, lang_prob, policy, providers, privacy_tag)]
    for name in ("hinglish_local", "whisper_local", "passthrough"):
        p = providers.get(name)
        if p and p not in order:
            order.append(p)
    last_err = None
    for provider in order:
        try:
            res = provider.transcribe(payload)
            res.segments = res.segments or []
            return res
        except Exception as e:  # timeout / unreachable -> next in chain
            last_err = e
    raise RuntimeError(f"all ASR providers failed: {last_err}")
