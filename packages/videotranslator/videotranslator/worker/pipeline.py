"""Worker processing pipeline (§7): orchestrates the processing ports.

The pipeline only talks to ports — which transcriber, cloner or renderer
sits behind them is a processing_profile decision made by the factory.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain.duration_fit import DurationPolicy, FitAction, decide_fit
from ..domain.enums import DomainError, ErrorCode
from ..domain.models import JobSpec, MediaType, TimedSegment
from ..ports import (
    AudioRenderer,
    DurationMatcher,
    MediaAssembler,
    ObjectStorage,
    ProgressReporter,
    SourceSeparator,
    SpeechTranscriber,
    TranslationProvider,
    VoiceCloner,
)


class WorkerPipeline:
    def __init__(
        self,
        storage: ObjectStorage,
        renderer: AudioRenderer,
        separator: SourceSeparator,
        transcriber: SpeechTranscriber,
        translator: TranslationProvider,
        cloner: VoiceCloner,
        matcher: DurationMatcher,
        assembler: MediaAssembler,
        progress: ProgressReporter,
        policy: DurationPolicy | None = None,
    ):
        self._storage = storage
        self._renderer = renderer
        self._separator = separator
        self._transcriber = transcriber
        self._translator = translator
        self._cloner = cloner
        self._matcher = matcher
        self._assembler = assembler
        self._progress = progress
        self._policy = policy or DurationPolicy()

    def run(self, spec: JobSpec, workdir: Path) -> Path:
        workdir.mkdir(parents=True, exist_ok=True)
        source = self._download(spec.input_uri, workdir / "input")

        self._progress.report("extract", 5)
        audio = self._renderer.extract_audio(source, workdir / "audio.wav")

        self._progress.report("separate", 15)
        vocals = self._separator.separate_vocals(audio, workdir)

        self._progress.report("transcribe", 30)
        segments = self._transcriber.transcribe(vocals, spec.source_language)
        if not segments:
            raise DomainError("no speech segments found", code=ErrorCode.BACKEND_FAILED)

        self._progress.report("translate", 45)
        base_durations = [s.end_ms - s.start_ms for s in segments]
        segments = self._translator.translate(segments, spec.target_language, base_durations)

        self._progress.report("synthesize", 60)
        fitted = self._synthesize_and_fit(segments, vocals, workdir)

        self._progress.report("assemble", 85)
        is_video = spec.output_uri.endswith(".mp4")
        output = workdir / ("result.mp4" if is_video else "result.mp3")
        self._assembler.assemble(source, fitted, None, output, MediaType.VIDEO if is_video else MediaType.AUDIO)

        self._progress.report("upload", 95)
        self._storage.upload(output, spec.output_uri.removeprefix("obj://"))
        self._progress.report("done", 100)
        return output

    def _synthesize_and_fit(self, segments: list[TimedSegment], vocals: Path, workdir: Path) -> list[tuple[int, Path]]:
        fitted: list[tuple[int, Path]] = []
        for i, segment in enumerate(segments):
            if not segment.translated_text:
                continue
            base = segment.end_ms - segment.start_ms
            gap = (segments[i + 1].start_ms - segment.end_ms) if i + 1 < len(segments) else 0
            reference = self._renderer.clip(vocals, segment.start_ms, segment.end_ms, workdir / f"ref_{i}.wav")

            attempts = 0
            text = segment.translated_text
            while True:
                generated = self._cloner.synthesize(text, reference, workdir / f"gen_{i}.wav")
                generated_ms = self._renderer.duration_ms(generated)
                decision = decide_fit(generated_ms, base, gap, attempts, self._policy)
                if decision.action == FitAction.COMPACT:
                    attempts += 1
                    shorter = self._translator.translate(
                        [TimedSegment(segment.index, segment.start_ms, segment.end_ms, segment.source_text)],
                        "",
                        [int(base * 0.8)],
                    )
                    text = shorter[0].translated_text or text
                    continue
                if decision.action == FitAction.FAIL:
                    raise DomainError(
                        f"segment {i} cannot fit after {attempts} compactions",
                        code=ErrorCode.DURATION_FIT_FAILED,
                    )
                out = self._matcher.fit(generated, base, gap, workdir / f"fit_{i}.wav")
                fitted.append((segment.start_ms, out))
                break
        return fitted

    def _download(self, uri: str, destination: Path) -> Path:
        key = uri.removeprefix("obj://")
        local = self._storage.local_path(key)
        if local is not None:
            return local
        return self._storage.download(key, destination)


class JsonProgressReporter:
    """Structured progress: atomic JSON file (§6.7) + stdout lines."""

    def __init__(self, progress_file: Path | None = None, output_object_key: str = ""):
        self._file = progress_file
        self._output_key = output_object_key

    def report(self, stage: str, percent: int) -> None:
        payload = {"stage": stage, "percent": percent, "output_object_key": self._output_key or None}
        print(json.dumps(payload), flush=True)
        if self._file is not None:
            tmp = self._file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._file)  # atomic publish
