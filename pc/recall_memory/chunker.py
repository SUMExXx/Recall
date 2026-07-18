"""Fixed-size token-window chunking — plan §5.

One rule everywhere: count tokens, cut at the window size, step back by the
overlap, move on. No drift detection, no merge rules. The only per-source
niceties are the cheap boundary preferences from the plan table:
  meeting  — never split a speaker attribution off from its line
  github   — prefer the nearest line boundary if one is close
  pdf      — prefer the nearest paragraph break if one is close
  image    — whole OCR block, single chunk
"""
from __future__ import annotations

from typing import Callable

from .config import CHUNK_SPECS, ChunkSpec
from .nmo import NMO, Chunk, Episode
from .tokenizer import Token
from .tokenizer import tokenize as _regex_tokenize

# A tokenizer is any callable text -> [Token] with char spans. Defaults to the
# regex tokenizer; the ingestor passes the active backend's model tokenizer so
# fixed windows line up with the Nomic seqlen graphs on the NPU (plan §5).
Tokenize = Callable[[str], list[Token]]

# How far back (fraction of window) we will move a cut to honor a cheap boundary.
_BOUNDARY_SLACK = 0.10


def _window_starts(n_tokens: int, spec: ChunkSpec) -> list[int]:
    stride = max(1, spec.size - spec.overlap)
    starts = list(range(0, max(1, n_tokens - spec.overlap), stride))
    if not starts:
        starts = [0]
    return starts


def _make_chunk(nmo: NMO, index: int, tokens: list[Token], text: str,
                lo: int, hi: int) -> Chunk:
    """Build a chunk over tokens[lo:hi] (hi exclusive) with exact char spans."""
    char_start, char_end = tokens[lo].start, tokens[hi - 1].end
    return Chunk(
        memory_id=nmo.memory_id, chunk_index=index, token_count=hi - lo,
        text=text[char_start:char_end], char_start=char_start, char_end=char_end,
    ).inherit(nmo)


def _prefer_boundary(text: str, tokens: list[Token], lo: int, hi: int,
                     sep: str) -> int:
    """Pull `hi` back to just after the last `sep` inside the slack region, if any."""
    slack = max(1, int(_BOUNDARY_SLACK * (hi - lo)))
    for j in range(hi - 1, max(lo, hi - 1 - slack), -1):
        gap = text[tokens[j - 1].end:tokens[j].start] if j > lo else ""
        if sep in gap:
            return j
    return hi


def chunk_document(nmo: NMO, tokenize: Tokenize = _regex_tokenize) -> list[Chunk]:
    """Fixed windows for github_repo / pdf / text / note; whole-block for image."""
    text = nmo.content
    if nmo.source_type == "image":
        tokens = tokenize(text)
        if not tokens:
            return []
        return [_make_chunk(nmo, 0, tokens, text, 0, len(tokens))]

    spec = CHUNK_SPECS[nmo.source_type]
    tokens = tokenize(text)
    if not tokens:
        return []
    sep = "\n\n" if nmo.source_type == "pdf" else (
        "\n" if nmo.source_type == "github_repo" else "")
    chunks: list[Chunk] = []
    for idx, lo in enumerate(_window_starts(len(tokens), spec)):
        hi = min(lo + spec.size, len(tokens))
        if sep and hi < len(tokens):
            hi = _prefer_boundary(text, tokens, lo, hi, sep)
        chunks.append(_make_chunk(nmo, idx, tokens, text, lo, hi))
        if hi >= len(tokens):
            break
    return chunks


def build_transcript(episodes: list[Episode]) -> tuple[str, list[tuple[int, int, Episode]]]:
    """Render episodes as 'Speaker: text' lines; return (text, [(start,end,ep)] spans)."""
    parts: list[str] = []
    spans: list[tuple[int, int, Episode]] = []
    pos = 0
    for ep in sorted(episodes, key=lambda e: e.t_start):
        line = f"{ep.speaker}: {ep.text}" if ep.speaker else ep.text
        parts.append(line)
        spans.append((pos, pos + len(line), ep))
        pos += len(line) + 1  # '\n'
    return "\n".join(parts), spans


def chunk_meeting(nmo: NMO, episodes: list[Episode],
                  tokenize: Tokenize = _regex_tokenize) -> list[Chunk]:
    """Fixed windows over the episode-ordered transcript.

    Speaker-attribution rule: a window never *starts* between a line's
    'Speaker:' label and that line's first content token — if a cut would land
    there, it snaps back to the line start so the label stays with its text.
    Each chunk records the episode_ids / time span / speaker_span it covers.
    """
    text, spans = build_transcript(episodes)
    tokens = tokenize(text)
    if not tokens:
        return []
    spec = CHUNK_SPECS["meeting"]

    # Map char offset -> owning line start, and mark each line's label region.
    line_start_of: dict[int, int] = {}
    label_end_of: dict[int, int] = {}   # line char_start -> char end of "Speaker:"
    for s, e, ep in spans:
        line_start_of[s] = s
        label_end_of[s] = s + (len(ep.speaker) + 1 if ep.speaker else 0)

    def line_of(char_pos: int) -> tuple[int, int, Episode] | None:
        for s, e, ep in spans:
            if s <= char_pos <= e:
                return (s, e, ep)
        return None

    # Token index of each line start, for snapping.
    tok_at_char = {t.start: i for i, t in enumerate(tokens)}

    def snap_start(lo: int) -> int:
        """If tokens[lo] sits inside a line's label region, snap to line start."""
        ln = line_of(tokens[lo].start)
        if ln is None:
            return lo
        s, _, _ = ln
        if tokens[lo].start < label_end_of.get(s, s) and s in tok_at_char:
            return tok_at_char[s]
        return lo

    chunks: list[Chunk] = []
    idx = 0
    for lo in _window_starts(len(tokens), spec):
        lo = snap_start(lo)
        hi = min(lo + spec.size, len(tokens))
        chunk = _make_chunk(nmo, idx, tokens, text, lo, hi)
        covered = [ep for s, e, ep in spans
                   if s < chunk.char_end and e > chunk.char_start]
        chunk.episode_ids = [ep.episode_id for ep in covered]
        if covered:
            chunk.t_start = min(ep.t_start for ep in covered)
            chunk.t_end = max(ep.t_end for ep in covered)
            chunk.speaker_span = sorted({ep.speaker for ep in covered if ep.speaker})
        chunks.append(chunk)
        idx += 1
        if hi >= len(tokens):
            break
    return chunks
