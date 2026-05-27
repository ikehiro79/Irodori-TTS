from __future__ import annotations

import re
import secrets
from dataclasses import replace
from pathlib import Path
from typing import Callable

import torch

from .inference_runtime import InferenceRuntime, SamplingRequest, SamplingResult, save_wav


_TERMINAL_PUNCTUATION = "".join(chr(code) for code in (0x3002, 0xFF01, 0xFF1F))
_TERMINAL_PUNCTUATION += "!?." + chr(0xFF0E)
_SOFT_BREAK_CHARS = "".join(chr(code) for code in (0x3001, 0xFF0C))
_SOFT_BREAK_CHARS += ",;" + chr(0xFF1B) + ":" + chr(0xFF1A) + " "
_SENTENCE_RE = re.compile(
    rf"[^{re.escape(_TERMINAL_PUNCTUATION)}]+[{re.escape(_TERMINAL_PUNCTUATION)}]*"
)


def split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """Split long text into stable TTS-sized chunks, preferring sentence boundaries."""
    source = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if source == "":
        return []
    if (max_chars <= 0 or len(source) <= max_chars) and "\n" not in source:
        return [source]

    chunks: list[str] = []
    current = ""
    for line in source.split("\n"):
        line = line.strip()
        if line == "":
            if current:
                chunks.append(current)
                current = ""
            continue
        for match in _SENTENCE_RE.finditer(line):
            unit = match.group(0).strip()
            if unit == "":
                continue
            for piece in _split_oversized_unit(unit, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    chunks.append(current)
                    current = piece
                elif current:
                    current = _join_text_chunks(current, piece)
                else:
                    current = piece
        if current:
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)
    return chunks or [source]


def _join_text_chunks(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if right[:1] in _TERMINAL_PUNCTUATION:
        return f"{left}{right}"
    if left[-1:].isascii() and left[-1:].isalnum() and right[:1].isascii() and right[:1].isalnum():
        return f"{left} {right}"
    return f"{left}{right}"


def _split_oversized_unit(unit: str, max_chars: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]

    pieces: list[str] = []
    rest = unit.strip()
    while len(rest) > max_chars:
        window = rest[:max_chars]
        split_at = max(window.rfind(ch) for ch in _SOFT_BREAK_CHARS)
        if split_at < max_chars // 2:
            split_at = max_chars
        else:
            split_at += 1
        piece = rest[:split_at].strip()
        if piece:
            pieces.append(piece)
        rest = rest[split_at:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def concatenate_audios(
    chunk_results: list[SamplingResult],
    *,
    silence_ms: int = 120,
) -> SamplingResult:
    if not chunk_results:
        raise ValueError("chunk_results must not be empty.")

    sample_rate = int(chunk_results[0].sample_rate)
    num_candidates = len(chunk_results[0].audios)
    if any(int(result.sample_rate) != sample_rate for result in chunk_results):
        raise ValueError("All chunks must use the same sample rate.")
    if any(len(result.audios) != num_candidates for result in chunk_results):
        raise ValueError("All chunks must have the same number of candidates.")

    silence_samples = max(0, int(sample_rate * max(0, int(silence_ms)) / 1000.0))
    merged: list[torch.Tensor] = []
    for candidate_idx in range(num_candidates):
        parts: list[torch.Tensor] = []
        for chunk_idx, result in enumerate(chunk_results):
            audio = result.audios[candidate_idx].float().cpu()
            parts.append(audio)
            if silence_samples > 0 and chunk_idx < len(chunk_results) - 1:
                parts.append(torch.zeros((audio.shape[0], silence_samples), dtype=audio.dtype))
        merged.append(torch.cat(parts, dim=-1))

    messages: list[str] = [
        f"info: long text was split into {len(chunk_results)} chunks and concatenated.",
    ]
    for chunk_idx, result in enumerate(chunk_results, start=1):
        messages.append(f"chunk[{chunk_idx}] seed_used: {result.used_seed}")
        messages.extend(f"chunk[{chunk_idx}] {msg}" for msg in result.messages)

    stage_timings: list[tuple[str, float]] = []
    total_to_decode = 0.0
    for chunk_idx, result in enumerate(chunk_results, start=1):
        stage_timings.extend(
            (f"chunk[{chunk_idx}].{name}", seconds) for name, seconds in result.stage_timings
        )
        total_to_decode += float(result.total_to_decode)

    return SamplingResult(
        audio=merged[0],
        audios=merged,
        sample_rate=sample_rate,
        stage_timings=stage_timings,
        total_to_decode=total_to_decode,
        used_seed=chunk_results[0].used_seed,
        messages=messages,
    )


def synthesize_long_text(
    runtime: InferenceRuntime,
    req: SamplingRequest,
    *,
    max_chars: int,
    silence_ms: int = 120,
    log_fn: Callable[[str], None] | None = None,
    log_prefix: str = "[long-text]",
) -> SamplingResult:
    chunks = split_text_for_tts(req.text, max_chars=max_chars)
    if len(chunks) <= 1:
        return runtime.synthesize(req, log_fn=log_fn)

    if log_fn is not None:
        log_fn(f"{log_prefix} split text into {len(chunks)} chunks (max_chars={max_chars})")

    if req.seed is None:
        base_seed = int(secrets.randbits(63))
        if log_fn is not None:
            log_fn(f"{log_prefix} using shared random seed {base_seed} for all chunks")
    else:
        base_seed = int(req.seed)

    anchored_req = req
    anchor_path: Path | None = None
    can_anchor_no_ref = bool(
        req.no_ref
        and req.ref_wav is None
        and req.ref_latent is None
        and getattr(runtime.model_cfg, "use_speaker_condition", False)
    )
    if can_anchor_no_ref:
        if log_fn is not None:
            log_fn(f"{log_prefix} generating reference anchor for consistent voice timbre")
        anchor_result = runtime.synthesize(
            replace(req, text=chunks[0], seed=base_seed),
            log_fn=log_fn,
        )
        anchor_path = save_wav(
            Path("gradio_outputs") / "long_text_anchors" / f"anchor_{base_seed}.wav",
            anchor_result.audio.float(),
            anchor_result.sample_rate,
        )
        anchored_req = replace(
            req,
            ref_wav=str(anchor_path),
            ref_latent=None,
            no_ref=False,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
        )

    chunk_results: list[SamplingResult] = []
    for idx, chunk in enumerate(chunks, start=1):
        if log_fn is not None:
            log_fn(f"{log_prefix} chunk {idx}/{len(chunks)} chars={len(chunk)}")
        chunk_req = replace(anchored_req, text=chunk, seed=base_seed)
        chunk_results.append(runtime.synthesize(chunk_req, log_fn=log_fn))

    result = concatenate_audios(chunk_results, silence_ms=silence_ms)
    result.used_seed = base_seed
    result.messages.insert(
        1,
        f"info: shared seed {base_seed} was used for every text chunk to keep voice timbre consistent.",
    )
    if anchor_path is not None:
        result.messages.insert(
            2,
            f"info: generated anchor reference audio was reused for all chunks: {anchor_path}",
        )
    elif req.no_ref and getattr(runtime.model_cfg, "use_speaker_condition", False):
        result.messages.insert(
            2,
            "warning: speaker anchor was not used; provide reference audio for stronger voice consistency.",
        )
    elif not getattr(runtime.model_cfg, "use_speaker_condition", False):
        result.messages.insert(
            2,
            "info: this checkpoint has no speaker reference conditioning; use a consistent caption/style prompt to reduce voice changes.",
        )
    if req.seconds is not None:
        result.messages.insert(
            1,
            "info: manual seconds is applied to each text chunk before concatenation.",
        )
    return result
