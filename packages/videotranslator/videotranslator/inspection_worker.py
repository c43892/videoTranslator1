"""CPU inspection worker: ``python -m videotranslator.inspection_worker spec.json``.

Runs ffprobe over the input and writes ``inspection_result.json`` for the
local backend (or uploads it to the result URI in cloud profiles) — §13.3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .adapters.ffmpeg import FFmpegMediaInspector
from .adapters.local_storage import LocalObjectStorage
from .domain.enums import DomainError
from .domain.models import InspectionSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    result_path = spec_path.parent / "inspection_result.json"
    spec = InspectionSpec.from_json_dict(json.loads(spec_path.read_text()))

    try:
        storage = LocalObjectStorage(os.environ.get("LOCAL_STORAGE_DIR", "./vt-data/objects"))
        key = spec.input_uri.removeprefix("obj://")
        local = storage.local_path(key) or storage.download(key, spec_path.parent / "input")
        result = FFmpegMediaInspector().inspect(local)
    except DomainError as exc:
        result_path.write_text(json.dumps({"ok": False, "error": f"{exc.code.value}: {exc}"}))
        return 1
    except Exception as exc:
        result_path.write_text(json.dumps({"ok": False, "error": str(exc)[:300]}))
        return 1

    result_path.write_text(json.dumps(result.to_json_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
