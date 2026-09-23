"""Document models (§11) and worker contracts.

Every stored document carries ``version`` for optimistic concurrency inside
the document store; on ``Job`` it is the public ``status_version`` (§8, §16).
Timestamps are epoch milliseconds to stay JSON- and SQLite-friendly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any, ClassVar

from .enums import (
    JobStatus,
    LedgerEntryType,
    OutboxStatus,
    PaymentEventStatus,
    RefundStatus,
)


def now_ms() -> int:
    return int(time.time() * 1000)


class MediaType(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


@dataclass
class Document:
    """Base for all persisted documents."""

    COLLECTION: ClassVar[str] = ""

    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class User(Document):
    COLLECTION: ClassVar[str] = "users"

    user_id: str = ""
    email: str = ""
    display_name: str = ""
    photo_url: str | None = None
    point_balance_units: int = 0
    status: str = "active"  # active | suspended
    billing_status: str = "clear"  # clear | hold
    created_at: int = 0
    updated_at: int = 0


@dataclass
class UploadSession(Document):
    COLLECTION: ClassVar[str] = "upload_sessions"

    STATUS_UPLOADING: ClassVar[str] = "uploading"
    STATUS_COMMITTED: ClassVar[str] = "committed"
    STATUS_COMPLETING: ClassVar[str] = "completing"
    STATUS_COMPLETED: ClassVar[str] = "completed"
    STATUS_EXPIRED: ClassVar[str] = "expired"

    upload_id: str = ""
    owner_user_id: str = ""
    reserved_job_id: str = ""
    object_key: str = ""
    status: str = STATUS_UPLOADING
    original_filename: str = ""
    declared_size_bytes: int = 0
    block_size_bytes: int = 8 * 1024 * 1024
    file_fingerprint: str = ""
    target_language: str = ""
    sas_expires_at: int = 0
    last_activity_at: int = 0
    created_at: int = 0
    completed_at: int | None = None


@dataclass
class Job(Document):
    COLLECTION: ClassVar[str] = "jobs"

    job_id: str = ""
    owner_user_id: str = ""
    original_filename: str = ""
    media_type: str = MediaType.VIDEO
    duration_ms: int = 0
    duration_probe_raw: str = ""
    target_language: str = ""
    status: JobStatus = JobStatus.UPLOADED
    status_version: int = 0
    attempt_number: int = 1
    retry_of_job_id: str | None = None
    input_owner_job_id: str | None = None
    retry_allowed: bool = False
    retry_expires_at: int | None = None
    inspection_attempt: int = 0
    inspection_backend_job_id: str | None = None
    inspection_result_object_key: str | None = None
    stage: str = ""
    progress_percent: int = 0
    warnings: list[str] = field(default_factory=list)
    input_object_key: str = ""
    output_object_key: str | None = None
    quoted_point_units: int = 0
    pricing_version: str = ""
    charged_ledger_entry_id: str | None = None
    cost_reservation_id: str | None = None
    estimated_gpu_seconds: int = 0
    max_runtime_seconds: int = 0
    execution_deadline_at: int | None = None
    backend: str = ""
    backend_job_id: str | None = None
    submit_idempotency_key: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    refund_status: RefundStatus = RefundStatus.NOT_APPLICABLE
    balance_returned_cents: int = 0
    balance_returned_at: int | None = None
    asset_cleanup_status: str = "pending"
    assets_deleted_at: int | None = None
    cancel_requested_at: int | None = None
    capacity_last_tick_at: int | None = None  # last capacity-counter decrement
    terms_version: str = ""
    media_rights_attested_at: int | None = None
    created_at: int = 0
    queued_at: int | None = None
    started_at: int | None = None
    completed_at: int | None = None
    input_expires_at: int | None = None
    output_expires_at: int | None = None
    # Note: the store's optimistic-concurrency token for Job is
    # ``status_version`` (see adapters/store.doc_version); the inherited
    # ``version`` field is unused on this model.


@dataclass
class LedgerEntry(Document):
    COLLECTION: ClassVar[str] = "ledger_entries"

    ledger_entry_id: str = ""
    user_id: str = ""
    delta_units: int = 0
    entry_type: LedgerEntryType = LedgerEntryType.PURCHASE
    job_id: str | None = None
    payment_id: str | None = None
    balance_after_units: int = 0
    idempotency_key: str = ""
    created_at: int = 0


@dataclass
class Payment(Document):
    COLLECTION: ClassVar[str] = "payments"

    STATUS_PENDING: ClassVar[str] = "pending"
    STATUS_SUCCEEDED: ClassVar[str] = "succeeded"
    STATUS_REVERSED: ClassVar[str] = "reversed"

    payment_id: str = ""
    user_id: str = ""
    provider: str = ""
    redirect_url: str = ""
    confirmation_mode: str = ""
    provider_payment_id: str | None = None
    provider_order_id: str | None = None
    provider_capture_id: str | None = None
    package_id: str = ""
    currency: str = "USD"
    amount_minor: int = 0
    point_units: int = 0
    status: str = STATUS_PENDING
    created_at: int = 0
    completed_at: int | None = None


@dataclass
class PaymentEvent(Document):
    COLLECTION: ClassVar[str] = "payment_events"

    event_key: str = ""  # <provider>:<event_id>, the natural idempotency key
    provider: str = ""
    event_id: str = ""
    event_type: str = ""
    processing_status: PaymentEventStatus = PaymentEventStatus.RECEIVED
    payment_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    attempt_count: int = 0
    max_attempts: int = 20
    next_attempt_at: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None
    failure_class: str | None = None
    last_error: str | None = None
    dead_lettered_at: int | None = None
    alert_status: str | None = None
    received_at: int = 0
    processed_at: int | None = None


@dataclass
class PricingConfig(Document):
    COLLECTION: ClassVar[str] = "pricing_configs"

    pricing_version: str = "job-v1"
    point_units_per_minute: int = 100
    minimum_point_units: int = 1
    rounding: str = "ceil_final_point_unit"
    active_from: int = 0


@dataclass
class JobOutbox(Document):
    COLLECTION: ClassVar[str] = "job_outbox"

    outbox_id: str = ""  # job-submit:<job_id>:<n>
    job_id: str = ""
    action: str = "submit"
    status: OutboxStatus = OutboxStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = 10
    next_attempt_at: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None
    backend_job_id: str | None = None
    backend_job_name: str | None = None  # deterministic name (azure_job_name in cloud)
    jobspec: dict[str, Any] = field(default_factory=dict)  # frozen JobSpec snapshot
    failure_class: str | None = None
    last_error: str | None = None
    dead_lettered_at: int | None = None
    alert_status: str | None = None
    created_at: int = 0
    completed_at: int | None = None


@dataclass
class InspectionOutbox(Document):
    COLLECTION: ClassVar[str] = "inspection_outbox"

    outbox_id: str = ""  # inspection:<job_id>:<attempt>
    job_id: str = ""
    inspection_attempt: int = 1
    status: OutboxStatus = OutboxStatus.PENDING
    attempt_count: int = 0
    max_attempts: int = 5
    next_attempt_at: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None
    backend_inspection_id: str | None = None
    backend_execution_id: str | None = None
    spec: dict[str, Any] = field(default_factory=dict)  # frozen InspectionSpec snapshot
    result_object_key: str | None = None
    failure_class: str | None = None
    last_error: str | None = None
    dead_lettered_at: int | None = None
    alert_status: str | None = None
    created_at: int = 0
    completed_at: int | None = None


@dataclass
class CostReservation(Document):
    COLLECTION: ClassVar[str] = "cost_reservations"

    STATUS_ACTIVE: ClassVar[str] = "active"
    STATUS_AWAITING_ACTUAL: ClassVar[str] = "awaiting_actual_cost"
    STATUS_SETTLED: ClassVar[str] = "settled"

    reservation_id: str = ""  # costres_<job_id>, unique per job
    job_id: str = ""
    currency: str = "USD"
    sku_region_price_version: str = ""
    hourly_rate_minor: int = 0
    estimated_gpu_seconds: int = 0
    max_runtime_seconds: int = 0
    reserved_amount_minor: int = 0
    status: str = STATUS_ACTIVE
    day_period: str = ""
    month_period: str = ""
    actual_amount_minor: int | None = None
    created_at: int = 0
    reconciled_at: int | None = None


@dataclass
class CostBudgetPeriod(Document):
    COLLECTION: ClassVar[str] = "cost_budget_periods"

    period_id: str = ""  # <scope>:<period>, e.g. "day:2026-07-22"
    limit_minor: int = 0
    reserved_minor: int = 0
    settled_minor: int = 0


@dataclass
class CapacityCounter(Document):
    """Global GPU backlog counter (§13.6); single doc ``capacity_counters/global``."""

    COLLECTION: ClassVar[str] = "capacity_counters"

    counter_id: str = "global"
    reserved_gpu_seconds: int = 0


@dataclass
class TopUpPackage:
    package_id: str
    currency: str
    amount_minor: int
    point_units: int
    pricing_version: str


@dataclass(frozen=True)
class JobQuote:
    duration_ms: int
    point_units: int
    pricing_version: str


@dataclass(frozen=True)
class TimedSegment:
    index: int
    start_ms: int
    end_ms: int
    source_text: str
    translated_text: str | None = None


@dataclass(frozen=True)
class JobSpec:
    """Immutable worker contract (§7.2); frozen in the charge transaction."""

    schema_version: int
    job_id: str
    attempt_number: int
    input_uri: str
    output_uri: str
    duration_ms: int
    target_language: str
    source_language: str | None
    processing_profile: str
    duration_policy_version: str
    max_runtime_seconds: int

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "JobSpec":
        return cls(**data)


@dataclass(frozen=True)
class InspectionSpec:
    job_id: str
    inspection_attempt: int
    input_uri: str
    result_uri: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "InspectionSpec":
        return cls(**data)


@dataclass(frozen=True)
class MediaInspectionResult:
    duration_ms: int
    duration_probe_raw: str
    media_type: str
    has_audio: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "MediaInspectionResult":
        return cls(**data)


@dataclass(frozen=True)
class BackendJobRef:
    backend_job_id: str
    backend_job_name: str


@dataclass(frozen=True)
class BackendStatus:
    """Unified backend execution state observed by the Reconciler."""

    state: str  # queued | provisioning | running | succeeded | failed | cancelled
    stage: str = ""
    progress_percent: int = 0
    error_code: str | None = None
    error_message: str | None = None
    output_object_key: str | None = None
    started_at: int | None = None
    actual_gpu_seconds: int | None = None
    warnings: list[str] = field(default_factory=list)
