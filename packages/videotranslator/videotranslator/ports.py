"""Ports: protocols between the application layer and the outside world (§6).

Infrastructure concerns are expressed twice, deliberately small:

* Service ports (identity, storage, payment, job/inspection backends,
  processing) mirror §6 one-to-one.
* A single document ``Store`` port with optimistic-concurrency transactions
  backs every Unit of Work; Firestore, SQLite and the in-memory test store
  are interchangeable implementations of the same six-method contract.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from .domain.models import (
    BackendJobRef,
    BackendStatus,
    Document,
    InspectionSpec,
    JobSpec,
    MediaInspectionResult,
    TimedSegment,
    TopUpPackage,
)

D = TypeVar("D", bound=Document)


# ---------------------------------------------------------------------------
# Document store (transactional persistence)
# ---------------------------------------------------------------------------


class Tx(Protocol):
    """One atomic unit of work; optimistic concurrency via document versions."""

    def get(self, kind: type[D], doc_id: str) -> D | None: ...

    def put(self, doc: Document, doc_id: str) -> None:
        """Write a document previously read in this tx; version must still match."""
        ...

    def insert(self, doc: Document, doc_id: str) -> None:
        """Create a new document; fails if the id already exists."""
        ...

    def delete(self, kind: type[D], doc_id: str) -> None: ...

    def query(
        self,
        kind: type[D],
        *,
        where: tuple[str, str, Any] | None = None,
        where_in: tuple[str, list[Any]] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[D]: ...


class Store(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        """Yield a Tx; commit on clean exit, roll back on exception.

        Raises ``StaleVersion`` on optimistic-concurrency conflict; callers
        may retry the whole block.
        """
        ...


# ---------------------------------------------------------------------------
# §6.1 Identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserIdentity:
    uid: str
    email: str = ""
    email_verified: bool = False
    display_name: str = ""
    photo_url: str | None = None


class IdentityVerifier(Protocol):
    def verify(self, bearer_token: str) -> UserIdentity: ...


# ---------------------------------------------------------------------------
# §6.2 Job / inspection backends
# ---------------------------------------------------------------------------


class JobBackend(Protocol):
    name: str

    def submit(self, spec: JobSpec, idempotency_key: str) -> BackendJobRef: ...

    def get_status(self, backend_job_id: str) -> BackendStatus: ...

    def cancel(self, backend_job_id: str) -> bool: ...


class MediaInspectionBackend(Protocol):
    name: str

    def submit(self, spec: InspectionSpec, idempotency_key: str) -> BackendJobRef: ...

    def get_status(self, backend_job_id: str) -> BackendStatus: ...

    def get_result(self, backend_job_id: str) -> MediaInspectionResult: ...


# ---------------------------------------------------------------------------
# §6.3 Object storage
# ---------------------------------------------------------------------------


class ObjectStorage(Protocol):
    def create_upload_url(self, object_key: str, expires_in: int) -> str: ...

    def create_download_url(self, object_key: str, expires_in: int) -> str: ...

    def download(self, object_key: str, destination: Path) -> Path: ...

    def upload(self, source: Path, object_key: str) -> str: ...

    def exists(self, object_key: str, *, expected_size: int | None = None) -> bool: ...

    def local_path(self, object_key: str) -> Path | None:
        """Filesystem path when the backend is local, else None (→ async inspection)."""
        ...

    def delete(self, object_key: str) -> None: ...


# ---------------------------------------------------------------------------
# §6.4 Payments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaymentSessionRequest:
    payment_id: str  # local id doubles as the provider idempotency key
    user_id: str
    package: TopUpPackage
    success_url: str
    cancel_url: str


@dataclass(frozen=True)
class PaymentSession:
    payment_id: str
    provider: str
    confirmation_mode: str  # webhook | server_capture
    redirect_url: str
    provider_order_id: str | None = None


@dataclass(frozen=True)
class PaymentEventData:
    provider: str
    event_id: str
    event_type: str
    payment_id: str | None
    provider_payment_id: str | None
    provider_capture_id: str | None
    amount_minor: int
    currency: str
    payload: dict


@dataclass(frozen=True)
class PaymentSnapshot:
    provider_payment_id: str
    status: str  # completed | pending | failed
    amount_minor: int
    currency: str
    user_id: str | None = None
    provider_capture_id: str | None = None
    payment_reference: str | None = None


class PaymentGateway(Protocol):
    provider: str

    def create_session(self, request: PaymentSessionRequest) -> PaymentSession: ...

    def capture(self, provider_order_id: str, idempotency_key: str) -> PaymentSnapshot: ...

    def verify_webhook(self, headers: Mapping[str, str], raw_body: bytes) -> PaymentEventData: ...

    def get_payment(self, provider_payment_id: str) -> PaymentSnapshot: ...

    def refund(self, provider_payment_id: str, amount_minor: int) -> bool: ...


# ---------------------------------------------------------------------------
# §6.6 Pricing / admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostAdmissionRequest:
    job_id: str
    duration_ms: int
    max_runtime_seconds: int
    now_ms: int


@dataclass(frozen=True)
class CostAdmissionDecision:
    allowed: bool
    reason: str = ""
    estimated_gpu_seconds: int = 0
    reserved_amount_minor: int = 0
    day_period: str = ""
    month_period: str = ""


class JobPricingPolicy(Protocol):
    def current_config(self): ...  # -> PricingConfig


class TopUpPricingPolicy(Protocol):
    def list_packages(self, currency: str) -> list[TopUpPackage]: ...

    def get_package(self, package_id: str) -> TopUpPackage: ...


# ---------------------------------------------------------------------------
# §6.7 Media processing (worker side)
# ---------------------------------------------------------------------------


class ProgressReporter(Protocol):
    def report(self, stage: str, percent: int) -> None: ...


class MediaInspector(Protocol):
    def inspect(self, media: Path) -> MediaInspectionResult: ...


class SourceSeparator(Protocol):
    def separate_vocals(self, audio: Path, out_dir: Path) -> Path: ...


class SpeechTranscriber(Protocol):
    def transcribe(self, audio: Path, source_language: str | None) -> list[TimedSegment]: ...


class TranslationProvider(Protocol):
    def translate(
        self,
        segments: list[TimedSegment],
        target_language: str,
        target_durations_ms: list[int] | None = None,
    ) -> list[TimedSegment]: ...


class VoiceCloner(Protocol):
    def synthesize(self, text: str, reference_audio: Path, destination: Path) -> Path: ...


class DurationMatcher(Protocol):
    def fit(self, generated: Path, base_duration_ms: int, gap_to_next_ms: int, out: Path) -> Path: ...


class AudioRenderer(Protocol):
    def extract_audio(self, media: Path, destination: Path) -> Path: ...

    def clip(self, audio: Path, start_ms: int, end_ms: int, destination: Path) -> Path: ...

    def duration_ms(self, audio: Path) -> int: ...


class MediaAssembler(Protocol):
    def assemble(
        self,
        source_media: Path,
        segments_audio: list[tuple[int, Path]],
        background: Path | None,
        output: Path,
        media_type: str,
    ) -> Path: ...
