"""Outbox dispatchers and the Payment Inbox processor (§8.4, §16).

Claim → external call → record, always: the claim happens in a transaction
that flips job state, so a user cancel racing a dispatch can only end in
exactly one of "cancelled + refunded" or "submitted" (§8.4, §16).
"""

from __future__ import annotations

import uuid

from ..docstore import Store
from ..domain.enums import (
    BackendError,
    ErrorCode,
    FailureClass,
    JobStatus,
    OutboxStatus,
    PaymentEventStatus,
    check_transition,
)
from ..domain.models import (
    InspectionOutbox,
    InspectionSpec,
    Job,
    JobOutbox,
    JobSpec,
    PaymentEvent,
    now_ms,
)
from ..domain.retry import backoff_ms, classify
from ..ports import JobBackend, MediaInspectionBackend
from .uow import BillingUnitOfWork, JobFundingUnitOfWork

JOB_OUTBOX_BASE_S, JOB_OUTBOX_CAP_S = 30, 1800
INSPECTION_BASE_S, INSPECTION_CAP_S = 15, 600
PAYMENT_BASE_S, PAYMENT_CAP_S = 30, 21600


class OutboxDispatcher:
    def __init__(
        self,
        store: Store,
        job_backend: JobBackend,
        inspection_backend: MediaInspectionBackend,
        funding: JobFundingUnitOfWork,
        *,
        retry_window_ms: int,
        max_retry_attempts: int = 3,
        lease_ms: int = 60_000,
        instance_id: str | None = None,
    ):
        self._store = store
        self._job_backend = job_backend
        self._inspection_backend = inspection_backend
        self._funding = funding
        self._retry_window_ms = retry_window_ms
        self._max_retry_attempts = max_retry_attempts
        self._lease_ms = lease_ms
        self._instance = instance_id or f"dispatcher-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # GPU job submissions
    # ------------------------------------------------------------------

    def dispatch_due_jobs(self, *, now: int = 0, limit: int = 10) -> int:
        now = now or now_ms()
        claimed = self._claim_job_outbox(now, limit)
        for outbox_id in claimed:
            self._submit_one(outbox_id, now)
        return len(claimed)

    def _claim_job_outbox(self, now: int, limit: int) -> list[str]:
        claimed: list[str] = []
        with self._store.transaction() as tx:
            due = [
                o
                for o in tx.query(
                    JobOutbox,
                    where_in=("status", [OutboxStatus.PENDING, OutboxStatus.PROCESSING]),
                )
                if o.next_attempt_at <= now and (o.lease_expires_at or 0) <= now
            ][:limit]
            for outbox in due:
                job = tx.get(Job, outbox.job_id)
                if job is None:
                    continue
                if job.status == JobStatus.CANCELLED:
                    outbox.status = OutboxStatus.CANCELLED
                    outbox.completed_at = now
                    tx.put(outbox, outbox.outbox_id)
                    continue
                first_claim = outbox.status == OutboxStatus.PENDING and job.status == JobStatus.QUEUED
                # Expired lease on a PROCESSING row: the previous owner died
                # mid-submit; re-run it under the same idempotency key (§11.7).
                takeover = outbox.status in (OutboxStatus.PROCESSING, OutboxStatus.PENDING) and job.status in (
                    JobStatus.SUBMITTING,
                    JobStatus.CANCELLING,
                )
                if not (first_claim or takeover):
                    continue
                if first_claim:
                    check_transition(job.status, JobStatus.SUBMITTING)
                    job.status = JobStatus.SUBMITTING  # queued → submitting (§8.4)
                    tx.put(job, job.job_id)
                outbox.status = OutboxStatus.PROCESSING
                outbox.lease_owner = self._instance
                outbox.lease_expires_at = now + self._lease_ms
                outbox.attempt_count += 1
                tx.put(outbox, outbox.outbox_id)
                claimed.append(outbox.outbox_id)
        return claimed

    def _submit_one(self, outbox_id: str, now: int) -> None:
        with self._store.transaction() as tx:
            outbox = tx.get(JobOutbox, outbox_id)
            spec = JobSpec.from_json_dict(outbox.jobspec)
            job_id = outbox.job_id
        try:
            ref = self._job_backend.submit(spec, idempotency_key=outbox_id)
        except Exception as exc:  # classification decides the retry path
            self._handle_job_submit_failure(outbox_id, job_id, exc, now)
            return
        cancel_after = False
        with self._store.transaction() as tx:
            outbox = tx.get(JobOutbox, outbox_id)
            outbox.status = OutboxStatus.COMPLETED
            outbox.backend_job_id = ref.backend_job_id
            outbox.backend_job_name = ref.backend_job_name
            outbox.completed_at = now_ms()
            tx.put(outbox, outbox_id)
            job = tx.get(Job, job_id)
            job.backend = self._job_backend.name
            job.backend_job_id = ref.backend_job_id
            if job.status == JobStatus.SUBMITTING:
                check_transition(job.status, JobStatus.PROVISIONING)
                job.status = JobStatus.PROVISIONING
            elif job.status == JobStatus.CANCELLING:
                cancel_after = True  # keep cancelling; never overwrite a cancel (§8.4)
            tx.put(job, job.job_id)
        if cancel_after:
            self._job_backend.cancel(ref.backend_job_id)

    def _handle_job_submit_failure(self, outbox_id: str, job_id: str, exc: Exception, now: int) -> None:
        failure_class = classify(exc)
        if failure_class == FailureClass.UNKNOWN_OUTCOME:
            # Ask the backend by its deterministic name before retrying (§16.1).
            found = self._job_backend.get_status(outbox_id)
            if found.state != "not_found":
                self._submit_one_success_unknown(outbox_id, job_id, found)
                return
            failure_class = FailureClass.RETRYABLE
        with self._store.transaction() as tx:
            outbox = tx.get(JobOutbox, outbox_id)
            outbox.failure_class = failure_class.value
            outbox.last_error = str(exc)[:300]
            outbox.lease_owner = None
            outbox.lease_expires_at = None
            exhausted = outbox.attempt_count >= outbox.max_attempts
            if failure_class == FailureClass.PERMANENT or exhausted:
                outbox.status = OutboxStatus.DEAD_LETTER
                outbox.dead_lettered_at = now
                outbox.alert_status = "pending"
                tx.put(outbox, outbox_id)
            else:
                outbox.status = OutboxStatus.PENDING
                outbox.next_attempt_at = now + backoff_ms(outbox.attempt_count, JOB_OUTBOX_BASE_S, JOB_OUTBOX_CAP_S)
                tx.put(outbox, outbox_id)
                return
        # dead-lettered: fail the job with a full refund (§16.2)
        self._funding.fail_and_refund(
            job_id,
            error_code=ErrorCode.SUBMIT_FAILED,
            error_message=str(exc),
            refund=True,
            retry_allowed=True,
            retry_window_ms=self._retry_window_ms,
            now=now,
        )

    def _submit_one_success_unknown(self, outbox_id: str, job_id: str, status) -> None:
        with self._store.transaction() as tx:
            outbox = tx.get(JobOutbox, outbox_id)
            outbox.status = OutboxStatus.COMPLETED
            outbox.backend_job_id = outbox.backend_job_id or outbox_id
            outbox.completed_at = now_ms()
            tx.put(outbox, outbox_id)
            job = tx.get(Job, job_id)
            job.backend = self._job_backend.name
            job.backend_job_id = outbox.backend_job_id
            if job.status == JobStatus.SUBMITTING:
                job.status = JobStatus.PROVISIONING
            tx.put(job, job.job_id)

    # ------------------------------------------------------------------
    # CPU inspections
    # ------------------------------------------------------------------

    def dispatch_due_inspections(self, *, now: int = 0, limit: int = 10) -> int:
        now = now or now_ms()
        claimed: list[str] = []
        with self._store.transaction() as tx:
            due = [
                o
                for o in tx.query(
                    InspectionOutbox,
                    where_in=("status", [OutboxStatus.PENDING, OutboxStatus.PROCESSING]),
                )
                if o.next_attempt_at <= now and (o.lease_expires_at or 0) <= now
            ][:limit]
            for outbox in due:
                job = tx.get(Job, outbox.job_id)
                if job is None or job.status != JobStatus.INSPECTING:
                    outbox.status = OutboxStatus.CANCELLED
                    outbox.completed_at = now
                    tx.put(outbox, outbox.outbox_id)
                    continue
                outbox.status = OutboxStatus.PROCESSING
                outbox.lease_owner = self._instance
                outbox.lease_expires_at = now + self._lease_ms
                outbox.attempt_count += 1
                tx.put(outbox, outbox.outbox_id)
                claimed.append(outbox.outbox_id)
        for outbox_id in claimed:
            self._submit_inspection(outbox_id, now)
        return len(claimed)

    def _submit_inspection(self, outbox_id: str, now: int) -> None:
        with self._store.transaction() as tx:
            outbox = tx.get(InspectionOutbox, outbox_id)
            spec = InspectionSpec.from_json_dict(outbox.spec)
            job_id = outbox.job_id
        try:
            ref = self._inspection_backend.submit(spec, idempotency_key=outbox_id)
        except Exception as exc:
            self._handle_inspection_failure(outbox_id, job_id, exc, now)
            return
        with self._store.transaction() as tx:
            outbox = tx.get(InspectionOutbox, outbox_id)
            outbox.status = OutboxStatus.COMPLETED
            outbox.backend_inspection_id = ref.backend_job_id
            outbox.backend_execution_id = ref.backend_job_name
            outbox.completed_at = now_ms()
            tx.put(outbox, outbox_id)
            job = tx.get(Job, job_id)
            job.inspection_backend_job_id = ref.backend_job_id
            tx.put(job, job.job_id)

    def _handle_inspection_failure(self, outbox_id: str, job_id: str, exc: Exception, now: int) -> None:
        failure_class = classify(exc)
        with self._store.transaction() as tx:
            outbox = tx.get(InspectionOutbox, outbox_id)
            outbox.failure_class = failure_class.value
            outbox.last_error = str(exc)[:300]
            outbox.lease_owner = None
            outbox.lease_expires_at = None
            exhausted = outbox.attempt_count >= outbox.max_attempts
            if failure_class == FailureClass.PERMANENT or exhausted:
                outbox.status = OutboxStatus.DEAD_LETTER
                outbox.dead_lettered_at = now
                outbox.alert_status = "pending"
                tx.put(outbox, outbox_id)
            else:
                outbox.status = OutboxStatus.PENDING
                outbox.next_attempt_at = now + backoff_ms(outbox.attempt_count, INSPECTION_BASE_S, INSPECTION_CAP_S)
                tx.put(outbox, outbox_id)
                return
        self._funding.fail_and_refund(
            job_id,
            error_code=ErrorCode.MEDIA_INSPECTION_FAILED,
            error_message=str(exc),
            refund=False,  # inspections never charge (§11.9)
            now=now,
        )


class PaymentInboxProcessor:
    """Lease-based processor for verified payment events (§10.2, §16.2)."""

    def __init__(self, store: Store, billing: BillingUnitOfWork, *, lease_ms: int = 60_000, instance_id: str | None = None):
        self._store = store
        self._billing = billing
        self._lease_ms = lease_ms
        self._instance = instance_id or f"inbox-{uuid.uuid4().hex[:8]}"

    def process_due(self, *, now: int = 0, limit: int = 20) -> int:
        now = now or now_ms()
        claimed: list[str] = []
        with self._store.transaction() as tx:
            due = [
                e
                for e in tx.query(
                    PaymentEvent,
                    where_in=(
                        "processing_status",
                        [PaymentEventStatus.RECEIVED, PaymentEventStatus.PROCESSING],
                    ),
                )
                if e.next_attempt_at <= now and (e.lease_expires_at or 0) <= now
            ][:limit]
            for event in due:
                event.processing_status = PaymentEventStatus.PROCESSING
                event.lease_owner = self._instance
                event.lease_expires_at = now + self._lease_ms
                event.attempt_count += 1
                tx.put(event, event.event_key)
                claimed.append(event.event_key)
        processed = 0
        for key in claimed:
            if self._process_one(key, now):
                processed += 1
        return processed

    def _process_one(self, event_key: str, now: int) -> bool:
        try:
            self._billing.apply_verified_payment(event_key, now=now)
            return True
        except Exception as exc:
            failure_class = classify(exc)
            # Domain validation failures (e.g. amount mismatch) are permanent.
            from ..domain.enums import DomainError

            if isinstance(exc, DomainError):
                failure_class = FailureClass.PERMANENT
            with self._store.transaction() as tx:
                event = tx.get(PaymentEvent, event_key)
                if event is None or event.processing_status == PaymentEventStatus.PROCESSED:
                    return False
                event.failure_class = failure_class.value
                event.last_error = str(exc)[:300]
                event.lease_owner = None
                event.lease_expires_at = None
                if failure_class == FailureClass.PERMANENT or event.attempt_count >= event.max_attempts:
                    event.processing_status = PaymentEventStatus.DEAD_LETTER
                    event.dead_lettered_at = now
                    event.alert_status = "pending"  # highest-priority payment alert (§16.2)
                else:
                    event.processing_status = PaymentEventStatus.RECEIVED
                    event.next_attempt_at = now + backoff_ms(event.attempt_count, PAYMENT_BASE_S, PAYMENT_CAP_S)
                tx.put(event, event_key)
            return False
