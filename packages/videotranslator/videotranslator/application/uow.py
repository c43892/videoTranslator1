"""Transactional units of work (§6.5, §8.3).

Every method performs ALL of its business writes inside one store
transaction — success means the whole thing happened, failure leaves no
partial state. Application code must never assemble charges, refunds or
payment postings out of individual repository calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import CostPolicy
from ..docstore import Store, Tx
from ..domain.enums import (
    ACTIVE_CHARGED_STATUSES,
    BackendError,
    CapacityUnavailable,
    DomainError,
    ErrorCode,
    InsufficientCredits,
    InvalidTransition,
    JobStatus,
    LedgerEntryType,
    NotFound,
    OutboxStatus,
    RefundStatus,
    check_transition,
)
from ..domain.models import (
    CapacityCounter,
    CostBudgetPeriod,
    CostReservation,
    Job,
    JobOutbox,
    JobSpec,
    LedgerEntry,
    Payment,
    PaymentEvent,
    PaymentEventStatus,
    PricingConfig,
    User,
    now_ms,
)
from ..domain.pricing import estimated_gpu_seconds, max_runtime_seconds, quote_job
from ..ids import new_id

# Deterministic ledger ids make insert-based idempotency natural (§16).
def ledger_id_for(idempotency_key: str) -> str:
    return "le_" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:20]


def job_charge_key(job_id: str) -> str:
    return f"job-charge:{job_id}"


def job_refund_key(job_id: str) -> str:
    return f"job-refund:{job_id}"


def purchase_key(payment_id: str) -> str:
    return f"purchase:{payment_id}"


def reversal_key(payment_id: str) -> str:
    return f"reversal:{payment_id}"


def _period_ids(now: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(now / 1000, tz=timezone.utc)
    return dt.strftime("day:%Y-%m-%d"), dt.strftime("month:%Y-%m")


@dataclass(frozen=True)
class CompleteInspectionCommand:
    job_id: str
    owner_user_id: str
    original_filename: str
    media_type: str
    target_language: str
    input_object_key: str
    duration_ms: int
    duration_probe_raw: str
    inspection_attempt: int
    terms_version: str = ""
    media_rights_attested_at: int = 0
    now: int = 0


@dataclass(frozen=True)
class ChargeResult:
    job: Job
    outcome: str  # queued | awaiting_credits | awaiting_capacity
    block_reason: str = ""


@dataclass(frozen=True)
class CancelResultView:
    job: Job
    outcome: str  # cancelled_refunded | cancelling | already_cancelled


class JobFundingUnitOfWork:
    """Owns the transaction boundary for charge/enqueue/cancel/fail (§6.5)."""

    def __init__(self, store: Store, pricing: PricingConfig, cost: CostPolicy, *, max_active_jobs_per_user: int = 1):
        self._store = store
        self._pricing = pricing
        self._cost = cost
        self._max_active = max_active_jobs_per_user

    # ------------------------------------------------------------------
    # shared charge path (§8.3 steps 1-8)
    # ------------------------------------------------------------------

    def _charge_tx(self, tx: Tx, job: Job, now: int) -> ChargeResult:
        """Run admission and, if admitted, charge + queue inside ``tx``.

        The caller guarantees ``job`` was read in this same transaction and
        is in a chargeable state (inspecting / awaiting_*).
        """
        user = tx.get(User, job.owner_user_id)
        if user is None:
            raise NotFound(f"user {job.owner_user_id}")
        if user.status == "suspended":
            raise DomainError("account suspended", code=ErrorCode.FORBIDDEN)

        if user.point_balance_units < job.quoted_point_units:
            self._move(tx, job, JobStatus.AWAITING_CREDITS)
            return ChargeResult(job, "awaiting_credits")

        reason = self._admission_block_reason(tx, job, now)
        if reason is not None:
            self._move(tx, job, JobStatus.AWAITING_CAPACITY)
            return ChargeResult(job, "awaiting_capacity", reason)

        self._reserve_and_charge(tx, job, user, now)
        return ChargeResult(job, "queued")

    def _admission_block_reason(self, tx: Tx, job: Job, now: int) -> str | None:
        cost = self._cost
        if not cost.gpu_starts_enabled:
            return "gpu_starts_disabled"
        active = tx.query(
            Job,
            where=("owner_user_id", "==", job.owner_user_id),
            where_in=("status", list(ACTIVE_CHARGED_STATUSES)),
        )
        if any(j.job_id != job.job_id for j in active):
            return "user_active_job_exists"
        counter = tx.get(CapacityCounter, "global")
        if counter is not None and counter.reserved_gpu_seconds >= cost.capacity_block_gpu_seconds:
            return "backlog_exceeded"
        if cost.enforce_budgets:
            day_id, month_id = _period_ids(now)
            reservation_amount = self._reservation_amount(job)
            for period_id in (day_id, month_id):
                period = tx.get(CostBudgetPeriod, period_id)
                if period is None:
                    period = CostBudgetPeriod(period_id=period_id, limit_minor=self._period_limit(period_id))
                    tx.insert(period, period_id)
                if period.settled_minor + period.reserved_minor + reservation_amount > period.limit_minor:
                    return "cost_budget_exceeded"
        return None

    def _period_limit(self, period_id: str) -> int:
        return self._cost.daily_budget_minor if period_id.startswith("day:") else self._cost.monthly_budget_minor

    def _reservation_amount(self, job: Job) -> int:
        # Worst-case reservation: max_runtime × hourly rate (§14.4).
        return -(-self._cost.hourly_rate_minor * job.max_runtime_seconds // 3600)

    def _reserve_and_charge(self, tx: Tx, job: Job, user: User, now: int) -> None:
        cost = self._cost
        job.max_runtime_seconds = job.max_runtime_seconds or max_runtime_seconds(job.duration_ms)
        job.estimated_gpu_seconds = job.estimated_gpu_seconds or estimated_gpu_seconds(
            job.duration_ms, cost.runtime_ratio, cost.provisioning_allowance_s
        )
        amount = self._reservation_amount(job)
        day_id, month_id = _period_ids(now)

        counter = tx.get(CapacityCounter, "global")
        if counter is None:
            counter = CapacityCounter(counter_id="global", reserved_gpu_seconds=0)
            tx.insert(counter, "global")
        counter.reserved_gpu_seconds += job.estimated_gpu_seconds
        tx.put(counter, "global")

        if cost.enforce_budgets:
            for period_id in (day_id, month_id):
                period = tx.get(CostBudgetPeriod, period_id)
                if period is None:
                    period = CostBudgetPeriod(period_id=period_id, limit_minor=self._period_limit(period_id))
                    tx.insert(period, period_id)
                period.reserved_minor += amount
                tx.put(period, period_id)
            reservation = CostReservation(
                reservation_id=f"costres_{job.job_id}",
                job_id=job.job_id,
                sku_region_price_version=cost.sku_region_price_version,
                hourly_rate_minor=cost.hourly_rate_minor,
                estimated_gpu_seconds=job.estimated_gpu_seconds,
                max_runtime_seconds=job.max_runtime_seconds,
                reserved_amount_minor=amount,
                status=CostReservation.STATUS_ACTIVE,
                day_period=day_id.split(":", 1)[1],
                month_period=month_id.split(":", 1)[1],
                created_at=now,
            )
            tx.insert(reservation, reservation.reservation_id)
            job.cost_reservation_id = reservation.reservation_id

        charge_key = job_charge_key(job.job_id)
        entry = LedgerEntry(
            ledger_entry_id=ledger_id_for(charge_key),
            user_id=user.user_id,
            delta_units=-job.quoted_point_units,
            entry_type=LedgerEntryType.JOB_CHARGE,
            job_id=job.job_id,
            balance_after_units=user.point_balance_units - job.quoted_point_units,
            idempotency_key=charge_key,
            created_at=now,
        )
        tx.insert(entry, entry.ledger_entry_id)
        user.point_balance_units = entry.balance_after_units
        user.updated_at = now
        tx.put(user, user.user_id)

        job.charged_ledger_entry_id = entry.ledger_entry_id
        job.execution_deadline_at = now + (cost.provisioning_allowance_s + job.max_runtime_seconds) * 1000
        job.capacity_last_tick_at = now
        job.queued_at = now
        self._move(tx, job, JobStatus.QUEUED)

        submit_key = f"job-submit:{job.job_id}:1"
        spec = self._build_jobspec(job)
        outbox = JobOutbox(
            outbox_id=submit_key,
            job_id=job.job_id,
            jobspec=spec.to_json_dict(),
            created_at=now,
        )
        tx.insert(outbox, submit_key)
        job.submit_idempotency_key = submit_key
        tx.put(job, job.job_id)

    def _build_jobspec(self, job: Job) -> JobSpec:
        return JobSpec(
            schema_version=2,
            job_id=job.job_id,
            attempt_number=job.attempt_number,
            input_uri=f"obj://{job.input_object_key}",
            output_uri=f"obj://users/{job.owner_user_id}/jobs/{job.job_id}/result"
                       + (".mp4" if job.media_type == "video" else ".mp3"),
            duration_ms=job.duration_ms,
            target_language=job.target_language,
            source_language=None,
            processing_profile="default-v1",
            duration_policy_version="duration-v1",
            max_runtime_seconds=job.max_runtime_seconds,
        )

    def _move(self, tx: Tx, job: Job, target: JobStatus) -> None:
        check_transition(job.status, target)
        job.status = target
        tx.put(job, job.job_id)

    # ------------------------------------------------------------------
    # public commands
    # ------------------------------------------------------------------

    def complete_inspection(self, command: CompleteInspectionCommand) -> ChargeResult:
        """Single completion entry for sync and async inspection (§3.2.1).

        Idempotent on the reserved job id: replays observe the job is no
        longer ``inspecting`` and return the current state without
        re-quoting or re-charging.
        """
        now = command.now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, command.job_id)
            if job is not None and job.status not in (JobStatus.INSPECTING, JobStatus.UPLOADED):
                outcome = "queued" if job.status == JobStatus.QUEUED else job.status.value
                return ChargeResult(job, outcome)
            if job is None:
                job = Job(
                    job_id=command.job_id,
                    owner_user_id=command.owner_user_id,
                    original_filename=command.original_filename,
                    media_type=command.media_type,
                    target_language=command.target_language,
                    input_object_key=command.input_object_key,
                    status=JobStatus.INSPECTING,
                    terms_version=command.terms_version,
                    media_rights_attested_at=command.media_rights_attested_at or now,
                    created_at=now,
                )
                tx.insert(job, job.job_id)
                job = tx.get(Job, job.job_id)
            if job.inspection_attempt not in (0, command.inspection_attempt):
                raise DomainError("stale inspection attempt", code=ErrorCode.INVALID_TRANSITION)
            quote = quote_job(self._pricing, command.duration_ms, command.media_type)  # may raise MEDIA_TOO_LONG
            job.inspection_attempt = command.inspection_attempt
            job.duration_ms = command.duration_ms
            job.duration_probe_raw = command.duration_probe_raw
            job.media_type = command.media_type
            job.quoted_point_units = quote.point_units
            job.pricing_version = quote.pricing_version
            tx.put(job, job.job_id)
            return self._charge_tx(tx, tx.get(Job, job.job_id), now)

    def fund_and_enqueue(self, job_id: str, *, now: int = 0) -> ChargeResult:
        """Manual "start" from awaiting_credits / awaiting_capacity (§8.3)."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, job_id)
            if job is None:
                raise NotFound(f"job {job_id}")
            if job.status in ACTIVE_CHARGED_STATUSES or job.status in (
                JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED
            ):
                return ChargeResult(job, job.status.value)  # idempotent replay
            if job.status == JobStatus.INSPECTING:
                raise InvalidTransition("job is still inspecting")
            user = tx.get(User, job.owner_user_id)
            if user is not None and user.point_balance_units < job.quoted_point_units:
                raise InsufficientCredits("balance below quote")
            return self._charge_tx(tx, job, now)

    def cancel(self, job_id: str, *, now: int = 0) -> CancelResultView:
        """§8.4: queued cancel refunds in the same tx; later states become cancelling."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, job_id)
            if job is None:
                raise NotFound(f"job {job_id}")
            if job.status == JobStatus.CANCELLED:
                return CancelResultView(job, "already_cancelled")
            if job.status == JobStatus.QUEUED:
                self._move(tx, job, JobStatus.CANCELLED)
                for outbox in tx.query(JobOutbox, where=("job_id", "==", job_id)):
                    if outbox.status in (OutboxStatus.PENDING, OutboxStatus.PROCESSING):
                        outbox.status = OutboxStatus.CANCELLED
                        outbox.completed_at = now
                        tx.put(outbox, outbox.outbox_id)
                job.refund_status = self._refund_tx(tx, job, now)
                self._release_capacity_and_settle(tx, job, now, actual_gpu_seconds=0)
                job.completed_at = now
                tx.put(job, job_id)
                return CancelResultView(tx.get(Job, job_id), "cancelled_refunded")
            if job.status in (JobStatus.SUBMITTING, JobStatus.PROVISIONING, JobStatus.RUNNING):
                self._move(tx, job, JobStatus.CANCELLING)
                job.cancel_requested_at = now
                tx.put(job, job.job_id)
                return CancelResultView(tx.get(Job, job_id), "cancelling")
            raise InvalidTransition(f"cannot cancel from {job.status}")

    def fail_and_refund(
        self,
        job_id: str,
        *,
        error_code: ErrorCode,
        error_message: str = "",
        refund: bool,
        retry_allowed: bool = False,
        retry_window_ms: int = 0,
        actual_gpu_seconds: int | None = None,
        now: int = 0,
    ) -> Job:
        """Terminal failure with optional refund and retry metadata (§4.4, §3.5)."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, job_id)
            if job is None:
                raise NotFound(f"job {job_id}")
            if job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED):
                return job
            self._move(tx, job, JobStatus.FAILED)
            job.error_code = error_code.value
            job.error_message = error_message[:500]
            job.completed_at = now
            job.refund_status = self._refund_tx(tx, job, now) if refund else RefundStatus.NOT_APPLICABLE
            job.retry_allowed = retry_allowed
            if retry_allowed:
                job.retry_expires_at = now + retry_window_ms
            self._release_capacity_and_settle(tx, job, now, actual_gpu_seconds=actual_gpu_seconds)
            tx.put(job, job.job_id)
            return tx.get(Job, job_id)

    def confirm_cancelled(self, job_id: str, *, ran: bool, actual_gpu_seconds: int | None = None, now: int = 0) -> Job:
        """Reconciler-confirmed backend cancellation (§8.4).

        Refund only when the worker never actually started (§4.4).
        """
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, job_id)
            if job is None:
                raise NotFound(f"job {job_id}")
            if job.status == JobStatus.CANCELLED:
                return job
            if job.status != JobStatus.CANCELLING:
                raise InvalidTransition(f"cannot confirm cancel from {job.status}")
            self._move(tx, job, JobStatus.CANCELLED)
            job.completed_at = now
            job.refund_status = RefundStatus.NOT_APPLICABLE if ran else self._refund_tx(tx, job, now)
            self._release_capacity_and_settle(tx, job, now, actual_gpu_seconds=actual_gpu_seconds)
            tx.put(job, job.job_id)
            return tx.get(Job, job_id)

    def settle_terminal(self, job_id: str, *, actual_gpu_seconds: int | None, now: int = 0) -> None:
        """Release remaining capacity and settle the cost reservation for a succeeded job."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = tx.get(Job, job_id)
            if job is None:
                raise NotFound(f"job {job_id}")
            self._release_capacity_and_settle(tx, job, now, actual_gpu_seconds=actual_gpu_seconds)
            tx.put(job, job_id)

    # ------------------------------------------------------------------
    # ledger helpers (all in-tx)
    # ------------------------------------------------------------------

    def _refund_tx(self, tx: Tx, job: Job, now: int) -> RefundStatus:
        if not job.charged_ledger_entry_id or job.quoted_point_units <= 0:
            return RefundStatus.NOT_APPLICABLE
        key = job_refund_key(job.job_id)
        existing = tx.get(LedgerEntry, ledger_id_for(key))
        if existing is not None:
            return RefundStatus.COMPLETED
        user = tx.get(User, job.owner_user_id)
        entry = LedgerEntry(
            ledger_entry_id=ledger_id_for(key),
            user_id=job.owner_user_id,
            delta_units=job.quoted_point_units,
            entry_type=LedgerEntryType.JOB_REFUND,
            job_id=job.job_id,
            balance_after_units=user.point_balance_units + job.quoted_point_units,
            idempotency_key=key,
            created_at=now,
        )
        tx.insert(entry, entry.ledger_entry_id)
        user.point_balance_units = entry.balance_after_units
        user.updated_at = now
        if user.point_balance_units >= 0 and user.billing_status == "hold":
            user.billing_status = "clear"
        tx.put(user, user.user_id)
        return RefundStatus.COMPLETED

    def _release_capacity_and_settle(self, tx: Tx, job: Job, now: int, *, actual_gpu_seconds: int | None) -> None:
        counter = tx.get(CapacityCounter, "global")
        if counter is not None and job.estimated_gpu_seconds > 0:
            counter.reserved_gpu_seconds = max(0, counter.reserved_gpu_seconds - job.estimated_gpu_seconds)
            tx.put(counter, "global")
        job.estimated_gpu_seconds = 0
        if not job.cost_reservation_id:
            return
        reservation = tx.get(CostReservation, job.cost_reservation_id)
        if reservation is None or reservation.status == CostReservation.STATUS_SETTLED:
            return
        if actual_gpu_seconds is None:
            actual_amount = reservation.reserved_amount_minor  # conservative: settle at full reservation
        else:
            actual_amount = -(-self._cost.hourly_rate_minor * actual_gpu_seconds // 3600)
        for period_id in (f"day:{reservation.day_period}", f"month:{reservation.month_period}"):
            period = tx.get(CostBudgetPeriod, period_id)
            if period is None:
                continue
            period.reserved_minor = max(0, period.reserved_minor - reservation.reserved_amount_minor)
            period.settled_minor += actual_amount
            tx.put(period, period_id)
        reservation.status = CostReservation.STATUS_SETTLED
        reservation.actual_amount_minor = actual_amount
        reservation.reconciled_at = now
        tx.put(reservation, reservation.reservation_id)


class BillingUnitOfWork:
    """Owns the transaction boundary for payment posting and reversal (§10)."""

    def __init__(self, store: Store):
        self._store = store

    def apply_verified_payment(self, event_key: str, *, now: int = 0) -> Payment:
        """Post one verified payment event: payment + purchase ledger + balance, atomically."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            event = tx.get(PaymentEvent, event_key)
            if event is None:
                raise NotFound(f"payment event {event_key}")
            if event.processing_status == PaymentEventStatus.PROCESSED:
                return tx.get(Payment, event.payment_id)
            payment = tx.get(Payment, event.payment_id)
            if payment is None:
                raise NotFound(f"payment {event.payment_id}")
            event.processing_status = PaymentEventStatus.PROCESSED
            event.processed_at = now
            tx.put(event, event.event_key)
            if payment.status == Payment.STATUS_SUCCEEDED:
                return payment  # replay via a second event for the same capture
            amount = int(event.payload.get("amount_minor", payment.amount_minor))
            currency = event.payload.get("currency", payment.currency)
            if amount != payment.amount_minor or currency != payment.currency:
                raise DomainError(
                    "amount/currency mismatch with package snapshot",
                    code=ErrorCode.PAYMENT_MISMATCH,
                )
            payment.status = Payment.STATUS_SUCCEEDED
            payment.completed_at = now
            payment.provider_payment_id = event.payload.get("provider_payment_id", payment.provider_payment_id)
            payment.provider_capture_id = event.payload.get("provider_capture_id", payment.provider_capture_id)
            tx.put(payment, payment.payment_id)

            user = tx.get(User, payment.user_id)
            key = purchase_key(payment.payment_id)
            entry = LedgerEntry(
                ledger_entry_id=ledger_id_for(key),
                user_id=payment.user_id,
                delta_units=payment.point_units,
                entry_type=LedgerEntryType.PURCHASE,
                payment_id=payment.payment_id,
                balance_after_units=user.point_balance_units + payment.point_units,
                idempotency_key=key,
                created_at=now,
            )
            tx.insert(entry, entry.ledger_entry_id)
            user.point_balance_units = entry.balance_after_units
            user.updated_at = now
            if user.point_balance_units >= 0 and user.billing_status == "hold":
                user.billing_status = "clear"
            tx.put(user, user.user_id)
            return payment

    def reverse_payment(self, payment_id: str, *, reason: str = "", now: int = 0) -> LedgerEntry:
        """Chargeback/refund: deduct points, allow negative balance, hold billing (§4.4)."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            payment = tx.get(Payment, payment_id)
            if payment is None:
                raise NotFound(f"payment {payment_id}")
            if payment.status != Payment.STATUS_SUCCEEDED:
                raise InvalidTransition(f"cannot reverse payment in status {payment.status}")
            payment.status = Payment.STATUS_REVERSED
            tx.put(payment, payment.payment_id)
            user = tx.get(User, payment.user_id)
            key = reversal_key(payment_id)
            existing = tx.get(LedgerEntry, ledger_id_for(key))
            if existing is not None:
                return existing
            entry = LedgerEntry(
                ledger_entry_id=ledger_id_for(key),
                user_id=payment.user_id,
                delta_units=-payment.point_units,
                entry_type=LedgerEntryType.PAYMENT_REVERSAL,
                payment_id=payment_id,
                balance_after_units=user.point_balance_units - payment.point_units,
                idempotency_key=key,
                created_at=now,
            )
            tx.insert(entry, entry.ledger_entry_id)
            user.point_balance_units = entry.balance_after_units
            user.updated_at = now
            if user.point_balance_units < 0:
                user.billing_status = "hold"
            tx.put(user, user.user_id)
            return entry
