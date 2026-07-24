"""Local subprocess backends (§13.2): SQLite-durable queue, leases, concurrency 1.

One ``_SubprocessQueue`` powers both ``LocalJobBackend`` (GPU worker CLI) and
``LocalCpuInspectionBackend`` (inspection worker CLI); the queue DB is the
only restart-recovery source for local scheduling.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..domain.models import (
    BackendJobRef,
    BackendStatus,
    InspectionSpec,
    JobSpec,
    MediaInspectionResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    pid INTEGER,
    progress_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
)
"""


class _SubprocessQueue:
    def __init__(self, db_path: str | Path, *, kind: str, module: str, max_concurrent: int = 1):
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._kind = kind
        self._module = module
        self._max_concurrent = max_concurrent
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # -- submit / status ----------------------------------------------------

    def submit(self, spec: JobSpec | InspectionSpec, idempotency_key: str) -> BackendJobRef:
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO jobs (id, kind, spec_json, state, created_at) VALUES (?, ?, ?, 'queued', ?)",
                (idempotency_key, self._kind, json.dumps(spec.to_json_dict()), time.time()),
            )
            self._db.commit()
        return BackendJobRef(idempotency_key, idempotency_key)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (backend_job_id,)).fetchone()
        if row is None:
            return BackendStatus(state="not_found")
        progress = json.loads(row["progress_json"]) if row["progress_json"] else {}
        started_ms = int(row["started_at"] * 1000) if row["started_at"] else None
        elapsed_s = int(row["finished_at"] - row["started_at"]) if row["started_at"] and row["finished_at"] else None
        return BackendStatus(
            state=row["state"],
            stage=progress.get("stage", ""),
            progress_percent=progress.get("percent", 0),
            error_message=row["error"],
            output_object_key=progress.get("output_object_key"),
            started_at=started_ms,
            actual_gpu_seconds=elapsed_s,
        )

    def cancel(self, backend_job_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (backend_job_id,)).fetchone()
            if row is None:
                return False
            if row["state"] == "queued":
                self._finish(backend_job_id, "cancelled")
                return True
            if row["state"] == "running" and row["pid"]:
                try:
                    os.kill(row["pid"], signal.SIGTERM)
                except OSError:
                    pass
                self._finish(backend_job_id, "cancelled")
                return True
            return False

    # -- scheduler pump (called by the local scheduler loop) -----------------

    def pump(self) -> int:
        """Start due jobs up to the concurrency cap and reap finished ones."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE kind = ? AND state IN ('queued', 'running') ORDER BY created_at",
                (self._kind,),
            ).fetchall()
            running = [r for r in rows if r["state"] == "running"]
            for row in running:
                self._reap(row)
            running = [r for r in rows if r["state"] == "running" and self._is_alive(r["pid"])]
            slots = max(0, self._max_concurrent - len(running))
            started = 0
            for row in [r for r in rows if r["state"] == "queued"][:slots]:
                self._spawn(row)
                started += 1
            return started

    def _spawn(self, row: sqlite3.Row) -> None:
        workdir = Path(tempfile.gettempdir()) / "vt-local" / row["id"].replace(":", "_")
        workdir.mkdir(parents=True, exist_ok=True)
        spec_file = workdir / "spec.json"
        spec_file.write_text(row["spec_json"])
        progress_file = workdir / "progress.json"
        log = open(workdir / "worker.log", "ab")
        proc = subprocess.Popen(
            [sys.executable, "-m", self._module, str(spec_file), "--progress-file", str(progress_file)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self._db.execute(
            "UPDATE jobs SET state = 'running', pid = ?, started_at = ? WHERE id = ?",
            (proc.pid, time.time(), row["id"]),
        )
        self._db.commit()

    def _reap(self, row: sqlite3.Row) -> None:
        pid = row["pid"]
        if pid and self._is_alive(pid):
            progress_file = Path(tempfile.gettempdir()) / "vt-local" / row["id"].replace(":", "_") / "progress.json"
            if progress_file.exists():
                self._db.execute(
                    "UPDATE jobs SET progress_json = ? WHERE id = ?",
                    (progress_file.read_text(), row["id"]),
                )
                self._db.commit()
            return
        workdir = Path(tempfile.gettempdir()) / "vt-local" / row["id"].replace(":", "_")
        result_file = workdir / "result.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            state = "succeeded" if result.get("ok") else "failed"
            error = result.get("error")
        else:
            state, error, result = "failed", "worker exited without a result", {}
        self._db.execute(
            "UPDATE jobs SET state = ?, error = ?, progress_json = ?, finished_at = ? WHERE id = ?",
            (state, error, json.dumps(result), time.time(), row["id"]),
        )
        self._db.commit()

    def _finish(self, job_id: str, state: str) -> None:
        self._db.execute(
            "UPDATE jobs SET state = ?, finished_at = ? WHERE id = ?", (state, time.time(), job_id)
        )
        self._db.commit()

    @staticmethod
    def _is_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


class LocalJobBackend:
    name = "local"

    def __init__(self, db_path: str | Path, *, max_concurrent: int = 1):
        self._queue = _SubprocessQueue(
            db_path, kind="gpu", module="videotranslator.worker.cli", max_concurrent=max_concurrent
        )

    def submit(self, spec: JobSpec, idempotency_key: str) -> BackendJobRef:
        return self._queue.submit(spec, idempotency_key)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        self._queue.pump()
        return self._queue.get_status(backend_job_id)

    def cancel(self, backend_job_id: str) -> bool:
        return self._queue.cancel(backend_job_id)


class LocalCpuInspectionBackend:
    name = "local-inspection"

    def __init__(self, db_path: str | Path, *, max_concurrent: int = 1):
        self._queue = _SubprocessQueue(
            db_path, kind="inspection", module="videotranslator.inspection_worker", max_concurrent=max_concurrent
        )
        self._results: dict[str, MediaInspectionResult] = {}

    def submit(self, spec: InspectionSpec, idempotency_key: str) -> BackendJobRef:
        self._queue.submit(spec, idempotency_key)
        return BackendJobRef(idempotency_key, idempotency_key)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        self._queue.pump()
        return self._queue.get_status(backend_job_id)

    def get_result(self, backend_job_id: str) -> MediaInspectionResult:
        result_file = (
            Path(tempfile.gettempdir()) / "vt-local" / backend_job_id.replace(":", "_") / "inspection_result.json"
        )
        return MediaInspectionResult.from_json_dict(json.loads(result_file.read_text()))
