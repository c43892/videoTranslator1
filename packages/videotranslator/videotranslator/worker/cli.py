"""Worker CLI: ``python -m videotranslator.worker.cli spec.json [--progress-file p]``.

Reads one immutable JobSpec, runs the pipeline, writes a machine-readable
``result.json`` next to the spec for the local scheduler to reap (§13.2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..adapters.fake import (
    FakeSourceSeparator,
    FakeSpeechTranscriber,
    FakeTranslationProvider,
    FakeVoiceCloner,
)
from ..adapters.ffmpeg import (
    FFmpegAudioRenderer,
    FFmpegDurationMatcher,
    FFmpegMediaAssembler,
)
from ..adapters.local_storage import LocalObjectStorage
from ..domain.enums import DomainError
from ..domain.models import JobSpec
from .pipeline import JsonProgressReporter, WorkerPipeline


def build_pipeline(progress_file: Path | None, output_key: str) -> WorkerPipeline:
    """``default-v1`` profile: FFmpeg plumbing + fake heavy models.

    Real Whisper/Demucs/DeepSeek/IndexTTS2 adapters slot in here without
    touching the pipeline (§6.7).
    """
    storage = LocalObjectStorage(os.environ.get("LOCAL_STORAGE_DIR", "./vt-data/objects"))
    return WorkerPipeline(
        storage=storage,
        renderer=FFmpegAudioRenderer(),
        separator=FakeSourceSeparator(),
        transcriber=FakeSpeechTranscriber(),
        translator=FakeTranslationProvider(),
        cloner=FakeVoiceCloner(),
        matcher=FFmpegDurationMatcher(),
        assembler=FFmpegMediaAssembler(),
        progress=JsonProgressReporter(progress_file, output_key),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--progress-file", default=None)
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    result_path = spec_path.parent / "result.json"
    spec = JobSpec.from_json_dict(json.loads(spec_path.read_text()))
    output_key = spec.output_uri.removeprefix("obj://")

    try:
        pipeline = build_pipeline(
            Path(args.progress_file) if args.progress_file else None, output_key
        )
        pipeline.run(spec, spec_path.parent / "work")
    except DomainError as exc:
        result_path.write_text(json.dumps({"ok": False, "error": f"{exc.code.value}: {exc}"}))
        return 1
    except Exception as exc:  # unexpected → backend failure, retryable path upstream
        result_path.write_text(json.dumps({"ok": False, "error": str(exc)[:300]}))
        return 1

    result_path.write_text(json.dumps({"ok": True, "output_object_key": output_key}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
