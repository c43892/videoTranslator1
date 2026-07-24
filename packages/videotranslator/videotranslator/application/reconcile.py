"""Job Reconciler: syncs backend truth into business state (§13.5).

Polling is the correctness mechanism — no cloud event is required. Also owns
the execution-deadline watchdog and capacity-counter decay.
"""

from __future__ import annotations

from ..config import CostPolicy
from ..docstore import Store
from ..domain.enums import ErrorCode, JobStatus, check_transition
from ..domain.models import CapacityCounter, Job, now_ms
from ..ports import JobBackend, MediaInspectionBackend, ObjectStorage
from .uow import CompleteInspectionCommand, JobFundingUnitOfWork

WATCHED_STATUSES = (
    JobStatus.INSPECTING,
    JobStatus.SUBMITTING,
    JobStatus.PROVISIONING,
    JobStatus.RUNNING,
    JobStatus.CANCELLING,
)


class JobReconciler:
    def __init__(
        self,
        store: Store,
        job_backend: JobBackend,
        inspection_backend: MediaInspectionBackend,
        storage: ObjectStorage,
        funding: JobFundingUnitOfWork,
        cost: CostPolicy,
        *,
        retry_window_ms: int,
    ):
        self._store = store
        self._job_backend = job_backend
        self._inspection_backend = inspection_backend
        self._storage = storage
        self._funding = funding
        self._cost = cost
        self._retry_window_ms = retry_window_ms

    def reconcile_once(self, *, now: int = 0) -> int:
        now = now or now_ms()
        with self._store.transaction() as tx:
            jobs = tx.query(Job, where_in=("status", list(WATCHED_STATUSES)))
        synced = 0
        for job in jobs:
            if job.status == JobStatus.INSPECTING and job.inspection_backend_job_id:
                synced += self._sync_inspection(job, now)
            elif job.backend_job_id and job.status != JobStatus.INSPECTING:
                synced += self._sync_execution(job, now)
            synced += self._enforce_deadline(job, now)
        return synced

    # ------------------------------------------------------------------

    def _sync_inspection(self, job: Job, now: int) -> int:
        status = self._inspection_backend.get_status(job.inspection_backend_job_id)
        if status.state in ("queued", "provisioning", "running"):
            return 0
        if status.state == "failed":
            self._funding.fail_and_refund(
                job.job_id,
                error_code=ErrorCode.MEDIA_INSPECTION_FAILED,
                error_message=status.error_message or "inspection failed",
                refund=False,
                now=now,
            )
            return 1
        if status.state != "succeeded":
            return 0
        result = self._inspection_backend.get_result(job.inspection_backend_job_id)
        self._funding.complete_inspection(
            CompleteInspectionCommand(
                job_id=job.job_id,
                owner_user_id=job.owner_user_id,
                original_filename=job.original_filename,
                media_type=result.media_type,
                target_language=job.target_language,
                input_object_key=job.input_object_key,
                duration_ms=result.duration_ms,
                duration_probe_raw=result.duration_probe_raw,
                inspection_attempt=job.inspection_attempt or 1,
                now=now,
            )
        )
        return 1

    def _sync_execution(self, job: Job, now: int) -> int:
        status = self._job_backend.get_status(job.backend_job_id)
        if status.state in ("queued", "not_found"):
            return 0
        if status.state in ("provisioning", "running"):
            self._mark_progress(job, status, now)
            return 1
        if status.state == "succeeded":
            output_key = status.output_object_key or _output_key(job)
            if not self._storage.exists(output_key):
                self._funding.fail_and_refund(
                    job.job_id,
                    error_code=ErrorCode.BACKEND_FAILED,
                    error_message="backend reported success but output object is missing",
                    refund=True,
                    retry_allowed=True,
                    retry_window_ms=self._retry_window_ms,
                    actual_gpu_seconds=status.actual_gpu_seconds,
                    now=now,
                )
                return 1
            self._mark_succeeded(job, output_key, status, now)
            return 1
        if status.state == "failed":
            self._funding.fail_and_refund(
                job.job_id,
                error_code=ErrorCode.BACKEND_FAILED,
                error_message=status.error_message or "backend failure",
                refund=True,
                retry_allowed=True,
                retry_window_ms=self._retry_window_ms,
                actual_gpu_seconds=status.actual_gpu_seconds,
                now=now,
            )
            return 1
        if status.state == "cancelled":
            ran = job.started_at is not None or status.started_at is not None
            if job.status == JobStatus.CANCELLING:
                self._funding.confirm_cancelled(
                    job.job_id, ran=ran, actual_gpu_seconds=status.actual_gpu_seconds, now=now
                )
            else:
                # backend cancelled without a user request → treat as system failure
                self._funding.fail_and_refund(
                    job.job_id,
                    error_code=ErrorCode.BACKEND_FAILED,
                    error_message="backend cancelled unexpectedly",
                    refund=True,
                    retry_allowed=True,
                    retry_window_ms=self._retry_window_ms,
                    actual_gpu_seconds=status.actual_gpu_seconds,
                    now=now,
                )
            return 1
        return 0

    def _mark_progress(self, job: Job, status, now: int) -> None:
        with self._store.transaction() as tx:
            current = tx.get(Job, job.job_id)
            if current.status not in (JobStatus.PROVISIONING, JobStatus.RUNNING, JobStatus.CANCELLING):
                return
            if status.state == "running" and current.status != JobStatus.RUNNING:
                check_transition(current.status, JobStatus.RUNNING)
                current.status = JobStatus.RUNNING
                current.started_at = current.started_at or status.started_at or now
            current.stage = status.stage or current.stage
            current.progress_percent = status.progress_percent or current.progress_percent
            self._tick_capacity(tx, current, now)
            tx.put(current, current.job_id)

    def _tick_capacity(self, tx, job: Job, now: int) -> None:
        """Decay the global backlog counter as the job consumes GPU seconds."""
        if job.capacity_last_tick_at is None or job.estimated_gpu_seconds <= 0:
            return
        elapsed = max(0, (now - job.capacity_last_tick_at) // 1000)
        if elapsed == 0:
            return
        delta = min(job.estimated_gpu_seconds, elapsed)
        counter = tx.get(CapacityCounter, "global")
        if counter is not None:
            counter.reserved_gpu_seconds = max(0, counter.reserved_gpu_seconds - delta)
            tx.put(counter, "global")
        job.estimated_gpu_seconds -= delta
        job.capacity_last_tick_at = now

    def _mark_succeeded(self, job: Job, output_key: str, status, now: int) -> None:
        output_retention_ms = 7 * 24 * 3_600_000
        with self._store.transaction() as tx:
            current = tx.get(Job, job.job_id)
            if current.status != JobStatus.RUNNING:
                return
            check_transition(current.status, JobStatus.SUCCEEDED)
            current.status = JobStatus.SUCCEEDED
            current.output_object_key = output_key
            current.progress_percent = 100
            current.completed_at = now
            current.output_expires_at = now + output_retention_ms
            self._tick_capacity(tx, current, now)
            tx.put(current, current.job_id)
        self._funding.settle_terminal(
            job.job_id, actual_gpu_seconds=status.actual_gpu_seconds, now=now
        )

    def _enforce_deadline(self, job: Job, now: int) -> int:
        if job.execution_deadline_at is None or now <= job.execution_deadline_at:
            return 0
        if job.status not in (JobStatus.PROVISIONING, JobStatus.RUNNING):
            return 0
        if job.backend_job_id:
            self._job_backend.cancel(job.backend_job_id)
        self._funding.fail_and_refund(
            job.job_id,
            error_code=ErrorCode.EXECUTION_TIMEOUT,
            error_message="execution deadline exceeded",
            refund=True,
            retry_allowed=True,
            retry_window_ms=self._retry_window_ms,
            now=now,
        )
        return 1


def _output_key(job: Job) -> str:
    suffix = ".mp4" if job.media_type == "video" else ".mp3"
    return f"users/{job.owner_user_id}/jobs/{job.job_id}/result{suffix}"
