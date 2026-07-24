"""Status enums, the job state machine, error codes and failure classes.

Pure domain: no SDK imports allowed in this package.
"""

from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    UPLOADED = "uploaded"
    INSPECTING = "inspecting"
    AWAITING_CREDITS = "awaiting_credits"
    AWAITING_CAPACITY = "awaiting_capacity"
    QUEUED = "queued"
    SUBMITTING = "submitting"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# §8 state machine. Terminal states have no outgoing transitions; `succeeded`
# is not terminal because results can still expire.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.UPLOADED: frozenset({JobStatus.INSPECTING}),
    JobStatus.INSPECTING: frozenset(
        {JobStatus.AWAITING_CREDITS, JobStatus.AWAITING_CAPACITY, JobStatus.QUEUED, JobStatus.FAILED}
    ),
    JobStatus.AWAITING_CREDITS: frozenset(
        {JobStatus.QUEUED, JobStatus.AWAITING_CAPACITY, JobStatus.EXPIRED}
    ),
    JobStatus.AWAITING_CAPACITY: frozenset(
        {JobStatus.QUEUED, JobStatus.AWAITING_CREDITS, JobStatus.EXPIRED}
    ),
    JobStatus.QUEUED: frozenset({JobStatus.SUBMITTING, JobStatus.CANCELLED}),
    JobStatus.SUBMITTING: frozenset(
        {JobStatus.PROVISIONING, JobStatus.FAILED, JobStatus.CANCELLING}
    ),
    JobStatus.PROVISIONING: frozenset(
        {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLING}
    ),
    JobStatus.CANCELLING: frozenset(
        {JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.RUNNING}
    ),
    JobStatus.RUNNING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLING}
    ),
    JobStatus.SUCCEEDED: frozenset({JobStatus.EXPIRED}),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
    JobStatus.EXPIRED: frozenset(),
}

TERMINAL_STATUSES = frozenset({JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED})

# Statuses holding a charged, in-flight execution (§13.6 per-user activity cap
# and cleanup protection).
ACTIVE_CHARGED_STATUSES = frozenset(
    {
        JobStatus.QUEUED,
        JobStatus.SUBMITTING,
        JobStatus.PROVISIONING,
        JobStatus.RUNNING,
        JobStatus.CANCELLING,
    }
)

# Statuses in which target_language may still be patched (§12.2).
LANGUAGE_MUTABLE_STATUSES = frozenset(
    {
        JobStatus.UPLOADED,
        JobStatus.INSPECTING,
        JobStatus.AWAITING_CREDITS,
        JobStatus.AWAITING_CAPACITY,
    }
)


def check_transition(current: JobStatus, target: JobStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(f"{current} -> {target} is not an allowed transition")


class RefundStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


class PaymentEventStatus(StrEnum):
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    DEAD_LETTER = "dead_letter"


class LedgerEntryType(StrEnum):
    PURCHASE = "purchase"
    JOB_CHARGE = "job_charge"
    JOB_REFUND = "job_refund"
    PAYMENT_REVERSAL = "payment_reversal"
    ADMIN_ADJUSTMENT = "admin_adjustment"
    PROMOTION = "promotion"


class FailureClass(StrEnum):
    RETRYABLE = "retryable"
    UNKNOWN_OUTCOME = "unknown_outcome"
    PERMANENT = "permanent"


class ErrorCode(StrEnum):
    MEDIA_INSPECTION_FAILED = "media_inspection_failed"
    MEDIA_UNSUPPORTED = "media_unsupported"
    MEDIA_TOO_LONG = "media_too_long"
    NO_AUDIO_TRACK = "no_audio_track"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    CAPACITY_UNAVAILABLE = "capacity_temporarily_unavailable"
    EMAIL_NOT_VERIFIED = "email_not_verified"
    STALE_JOB_VERSION = "stale_job_version"
    INVALID_TRANSITION = "invalid_transition"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    DURATION_FIT_FAILED = "duration_fit_failed"
    SUBMIT_FAILED = "submit_failed"
    BACKEND_FAILED = "backend_failed"
    EXECUTION_TIMEOUT = "execution_timeout"
    UPLOAD_INCOMPLETE = "upload_incomplete"
    PAYMENT_MISMATCH = "payment_mismatch"
    RETRY_NOT_ALLOWED = "retry_not_allowed"
    RETRY_EXHAUSTED = "retry_exhausted"


class DomainError(Exception):
    """Base for expected business failures; carries a stable error code."""

    code: ErrorCode = ErrorCode.BACKEND_FAILED

    def __init__(self, message: str = "", *, code: ErrorCode | None = None):
        super().__init__(message or (code or self.code).value)
        if code is not None:
            self.code = code


class InvalidTransition(DomainError):
    code = ErrorCode.INVALID_TRANSITION


class NotFound(DomainError):
    code = ErrorCode.NOT_FOUND


class Forbidden(DomainError):
    code = ErrorCode.FORBIDDEN


class InsufficientCredits(DomainError):
    code = ErrorCode.INSUFFICIENT_CREDITS


class CapacityUnavailable(DomainError):
    code = ErrorCode.CAPACITY_UNAVAILABLE


class StaleVersion(DomainError):
    """Optimistic-concurrency conflict inside the document store."""

    code = ErrorCode.STALE_JOB_VERSION


class EmailNotVerified(DomainError):
    code = ErrorCode.EMAIL_NOT_VERIFIED


class BackendError(Exception):
    """Failure of an external backend call, pre-classified per §16.1.

    Unclassified exceptions are treated as ``retryable`` by default — never
    silently dropped as permanent (§16.1).
    """

    def __init__(self, message: str, *, failure_class: FailureClass = FailureClass.RETRYABLE):
        super().__init__(message)
        self.failure_class = failure_class
