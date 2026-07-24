"""FFmpeg/ffprobe-backed processing adapters (§6.7 first-version mapping).

Works today on any machine with the ffmpeg binaries; heavier models
(Whisper, Demucs, IndexTTS2) have their own lazily-imported adapters.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..domain.duration_fit import DurationPolicy, FitAction, decide_fit
from ..domain.enums import DomainError, ErrorCode
from ..domain.models import MediaInspectionResult, MediaType
from ..domain.pricing import probe_duration_to_ms


def _run(cmd: list[str], *, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as exc:
        raise DomainError(f"ffmpeg/ffprobe not installed ({what})", code=ErrorCode.BACKEND_FAILED) from exc
    if proc.returncode != 0:
        raise DomainError(f"{what} failed: {proc.stderr[-300:]}", code=ErrorCode.BACKEND_FAILED)
    return proc


def ffprobe_json(path: Path) -> dict:
    proc = _run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        what="ffprobe",
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise DomainError("corrupt media: ffprobe produced no metadata", code=ErrorCode.MEDIA_INSPECTION_FAILED) from exc


class FFmpegMediaInspector:
    """Trusted server-side probe (§12.2): duration, type, audio presence."""

    def inspect(self, media: Path) -> MediaInspectionResult:
        try:
            info = ffprobe_json(media)
        except DomainError as exc:
            if exc.code == ErrorCode.BACKEND_FAILED:
                raise DomainError(str(exc), code=ErrorCode.MEDIA_INSPECTION_FAILED) from exc
            raise
        raw = info.get("format", {}).get("duration")
        if raw is None:
            raise DomainError("no duration in media", code=ErrorCode.MEDIA_INSPECTION_FAILED)
        streams = info.get("streams", [])
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        if not has_audio:
            raise DomainError("no audio track", code=ErrorCode.NO_AUDIO_TRACK)
        has_video = any(s.get("codec_type") == "video" for s in streams)
        return MediaInspectionResult(
            duration_ms=probe_duration_to_ms(raw),
            duration_probe_raw=raw,
            media_type=MediaType.VIDEO if has_video else MediaType.AUDIO,
            has_audio=True,
        )


class FFmpegAudioRenderer:
    def extract_audio(self, media: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            ["ffmpeg", "-y", "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(destination)],
            what="audio extraction",
        )
        return destination

    def clip(self, audio: Path, start_ms: int, end_ms: int, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start_ms / 1000:.3f}",
                "-to", f"{end_ms / 1000:.3f}",
                "-i", str(audio),
                "-ac", "1", "-ar", "16000",
                str(destination),
            ],
            what="audio clip",
        )
        return destination

    def duration_ms(self, audio: Path) -> int:
        raw = ffprobe_json(audio).get("format", {}).get("duration")
        if raw is None:
            raise DomainError("no duration", code=ErrorCode.BACKEND_FAILED)
        return probe_duration_to_ms(raw)


class FFmpegDurationMatcher:
    """Fits generated speech into the timeline per the v1 policy (§7.1).

    ``atempo`` is always generated/available — the legacy AudioStitcher bug
    passed target/generated and slowed long speech down even further.
    """

    def __init__(self, policy: DurationPolicy | None = None):
        self._policy = policy or DurationPolicy()
        self._renderer = FFmpegAudioRenderer()

    def fit(self, generated: Path, base_duration_ms: int, gap_to_next_ms: int, out: Path) -> Path:
        generated_ms = self._renderer.duration_ms(generated)
        decision = decide_fit(generated_ms, base_duration_ms, gap_to_next_ms, 0, self._policy)
        out.parent.mkdir(parents=True, exist_ok=True)

        if decision.action in (FitAction.PAD, FitAction.NATURAL):
            filters = f"apad,atrim=0:{base_duration_ms / 1000:.3f}"
        elif decision.action == FitAction.BORROW:
            filters = f"atrim=0:{decision.available_duration_ms / 1000:.3f}"
        elif decision.action == FitAction.SPEED_UP:
            filters = f"{self._atempo_chain(decision.atempo)},atrim=0:{decision.available_duration_ms / 1000:.3f}"
        else:  # COMPACT / FAIL are pipeline-level decisions, not renderable here
            raise DomainError("generated audio exceeds soft tempo limit", code=ErrorCode.DURATION_FIT_FAILED)

        _run(
            ["ffmpeg", "-y", "-i", str(generated), "-af", filters, "-ac", "1", "-ar", "16000", str(out)],
            what="duration fit",
        )
        return out

    @staticmethod
    def _atempo_chain(ratio: float) -> str:
        # One atempo filter accepts 0.5–100; chain anyway for extreme ratios.
        parts: list[str] = []
        remaining = ratio
        while remaining > 4.0:
            parts.append("atempo=4.0")
            remaining /= 4.0
        parts.append(f"atempo={remaining:.5f}")
        return ",".join(parts)


class FFmpegMediaAssembler:
    """Timeline mix: fitted clips anchored at start_ms over background (§7)."""

    def assemble(self, source_media, segments_audio, background, output, media_type):
        output.parent.mkdir(parents=True, exist_ok=True)
        if not segments_audio:
            shutil.copyfile(source_media, output)
            return output

        inputs: list[str] = []
        for _, clip in segments_audio:
            inputs += ["-i", str(clip)]
        if background is not None:
            inputs += ["-i", str(background)]

        delays = "".join(
            f"[{i}:a]adelay={start}|{start}[d{i}];" for i, (start, _) in enumerate(segments_audio)
        )
        labels = "".join(f"[d{i}]" for i in range(len(segments_audio)))
        if background is not None:
            labels += f"[{len(segments_audio)}:a]"
        mix = f"{delays}{labels}amix=inputs={len(segments_audio) + (1 if background else 0)}:normalize=0[aout]"

        if media_type == MediaType.VIDEO:
            cmd = ["ffmpeg", "-y", "-i", str(source_media), *inputs,
                   "-filter_complex", mix, "-map", "0:v", "-map", "[aout]", "-c:v", "copy", str(output)]
        else:
            cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", mix, "-map", "[aout]", str(output)]
        _run(cmd, what="final assembly")
        return output
