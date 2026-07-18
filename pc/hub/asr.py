"""ASRProvider workers — plan §2's interface, hub-side.

Providers:
  FasterWhisperProvider   in-process Whisper (faster-whisper, CPU int8) — the
                          zero-setup default on the laptop; no external server
  WhisperLocalProvider    posts WAV bytes to a whisper.cpp-contract /inference
                          endpoint (event PC: ORT-QNN Whisper-Base-En server)
  PassthroughProvider     dev shortcut for clients that send text utterances
  HinglishLocalProvider   BYOM Oriserve slot (falls back to Whisper here)
  SarvamCloudProvider     policy-gated opt-in; NEVER called unless the policy
                          engine allows it (3s timeout -> Hinglish -> Whisper)

`make_local_whisper(cfg)` picks in-process vs HTTP per `RECALL_ASR_MODE`.
"""
from __future__ import annotations

import io
import logging
import math
import wave
from dataclasses import dataclass, field

from recall_memory.tracing import step

log = logging.getLogger("recall.asr")


def _collapse_hallucination_loop(text: str, min_repeats: int = 4) -> str:
    """Detect Whisper's local infinite-repetition loops on silence / noise
    ("Yeah. Yeah. Yeah. ...") and collapse them to a single word. Returns text."""
    import re
    for span in (1, 2, 3):
        if span == 1:
            pattern = r"\b(\w+)([.,!?…]*\s+\1){" + str(min_repeats - 1) + r",}"
        elif span == 2:
            pattern = r"\b(\w+[.,!?…]*\s+\w+)([.,!?…]*\s+\1){" + str(min_repeats - 1) + r",}"
        else:
            pattern = r"\b(\w+[.,!?…]*\s+\w+[.,!?…]*\s+\w+)([.,!?…]*\s+\1){" + str(min_repeats - 1) + r",}"
        text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
    return text.strip()


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


class FasterWhisperProvider:
    """In-process Whisper via faster-whisper (CTranslate2, CPU int8).

    WAV bytes in, ASRResult out — no external server to start. The model
    (~74 MB for base.en) downloads to the HF cache on first transcription.
    Keeps name "whisper_local" so provenance metadata stays consistent.
    """

    name = "whisper_local"

    def __init__(self, model_name: str = "base.en", beam_size: int = 5,
                initial_prompt: str = ""):
        self.model_name = model_name
        # beam_size was hardcoded to 1 (greedy) here — the single biggest
        # cause of mangled domain-term transcription ("FDS5 keyword index"
        # for "FAISS keyword index", etc.); 5 is faster-whisper's own
        # library default and materially more accurate for modest CPU cost.
        self.beam_size = beam_size
        # Vocabulary bias for jargon Whisper otherwise guesses phonetically.
        self.initial_prompt = initial_prompt or None
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log.info("loading whisper model %s (downloads to the HF cache on "
                     "first use)…", self.model_name)
            self._model = WhisperModel(self.model_name, device="cpu",
                                       compute_type="int8")
            log.info("whisper model %s ready", self.model_name)
        return self._model

    def transcribe(self, payload: dict) -> ASRResult:
        """payload = {audio: bytes (wav, 16 kHz mono pcm16)}."""
        import numpy as np
        with wave.open(io.BytesIO(payload["audio"]), "rb") as w:
            pcm = w.readframes(w.getnframes())
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lang = "en" if self.model_name.endswith(".en") else None
        segs_iter, info = self._load().transcribe(
            audio, beam_size=self.beam_size, language=lang,
            initial_prompt=self.initial_prompt,
            # vad_filter: skip non-speech — without it Whisper hallucinates
            # text ("Thanks for watching") on silent windows, becoming junk
            # memories. condition_on_previous_text=False: don't feed prior
            # segments back as decoding context — that feedback loop is what
            # produces "Yeah. Yeah. Yeah. ..." repetition runs on noisy audio.
            # no_repeat_ngram_size: hard decoding-level ban on repeating the
            # same 3-gram, a second independent guard against the same loop.
            vad_filter=True, condition_on_previous_text=False,
            no_repeat_ngram_size=3)
        segments, texts, logprobs = [], [], []
        for s in segs_iter:
            segments.append({"t_start": s.start, "t_end": s.end, "text": s.text})
            texts.append(s.text)
            if s.avg_logprob is not None:
                logprobs.append(s.avg_logprob)
        conf = (min(1.0, math.exp(sum(logprobs) / len(logprobs)))
                if logprobs else 0.9)
        text = _collapse_hallucination_loop("".join(texts).strip())
        return ASRResult(text=text, lang=getattr(info, "language", "en") or "en",
                         segments=segments, confidence=conf)

    def capabilities(self) -> dict:
        return {"langs": ["en"], "word_ts": False, "diarization": False,
                "offline": True, "codemix": False}


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
    """Sarvam Saaras v3 — policy-gated opt-in cloud ASR (plan §2). NEVER
    invoked unless PolicyEngine.is_cloud_allowed() says so; `flush_audio` in
    hub/app.py only reaches for this after local Whisper's own language ID
    detects Indic audio, and wraps the call with a short timeout that falls
    back to HinglishLocal/WhisperLocal on any failure.

    REST contract (docs.sarvam.ai/api-reference/speech-to-text/transcribe):
      POST {endpoint}, multipart/form-data, header `api-subscription-key`,
      fields file/model/language_code/mode. Response JSON has `transcript`,
      `language_code`, `language_probability`, optional `timestamps`.
    """
    name = "sarvam_cloud"

    def __init__(self, api_key: str = "", model: str = "saaras:v3",
                language_code: str = "unknown", mode: str = "transcribe",
                timeout: float = 3.0,
                endpoint: str = "https://api.sarvam.ai/speech-to-text"):
        self.api_key = api_key
        self.model = model
        self.language_code = language_code
        self.mode = mode
        self.timeout = timeout
        self.endpoint = endpoint

    def transcribe(self, payload: dict) -> ASRResult:
        if not self.api_key:
            raise RuntimeError("sarvam_cloud has no RECALL_SARVAM_API_KEY configured")
        import requests
        with step("asr:sarvam_http_call", model=self.model,
                  language_code=self.language_code) as s:
            r = requests.post(
                self.endpoint,
                headers={"api-subscription-key": self.api_key},
                files={"file": ("audio.wav", payload["audio"], "audio/wav")},
                data={"model": self.model, "language_code": self.language_code,
                      "mode": self.mode},
                timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            s.detail(response_language=data.get("language_code"),
                     transcript=data.get("transcript"))
        ts = data.get("timestamps") or {}
        words = ts.get("words") or []
        starts = ts.get("start_time_seconds") or []
        ends = ts.get("end_time_seconds") or []
        segments = [{"t_start": st, "t_end": en, "text": w}
                    for w, st, en in zip(words, starts, ends)]
        return ASRResult(
            text=(data.get("transcript") or "").strip(),
            lang=data.get("language_code") or self.language_code,
            segments=segments,
            confidence=float(data.get("language_probability") or 0.85))

    def capabilities(self) -> dict:
        return {"langs": ["hi", "en", "+21 indic"], "word_ts": True,
                "diarization": True, "offline": False, "codemix": True}


def make_local_whisper(cfg):
    """Resolve the whisper_local provider per cfg.asr_mode.

    auto     in-process faster-whisper when the package is installed,
             else the HTTP contract at cfg.whisper_url
    embedded require in-process (clear error if faster-whisper is missing)
    http     always the external server (event PC: ORT-QNN Whisper)
    """
    mode = getattr(cfg, "asr_mode", "auto")
    if mode in ("auto", "embedded"):
        try:
            import faster_whisper  # noqa: F401 — availability probe only
            return FasterWhisperProvider(
                cfg.whisper_model, beam_size=getattr(cfg, "whisper_beam_size", 5),
                initial_prompt=getattr(cfg, "whisper_initial_prompt", ""))
        except ImportError:
            if mode == "embedded":
                raise RuntimeError(
                    "RECALL_ASR_MODE=embedded needs the in-process ASR: "
                    'pip install -e ".[asr]"')
    return WhisperLocalProvider(cfg.whisper_url)


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
        with step(f"asr:try_{provider.name}") as s:
            try:
                res = provider.transcribe(payload)
                res.segments = res.segments or []
                s.detail(ok=True, text=res.text[:200], confidence=res.confidence)
                return res
            except Exception as e:   # timeout / unreachable -> next in chain
                s.detail(ok=False, error=str(e))
                last_err = e
    raise RuntimeError(f"all ASR providers failed: {last_err}")
