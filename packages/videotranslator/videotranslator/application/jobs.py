"""Job-facing application service: uploads, start/cancel/retry, downloads (§12.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..docstore import Store
from ..domain.enums import (
    ACTIVE_CHARGED_STATUSES,
    DomainError,
    ErrorCode,
    Forbidden,
    InvalidTransition,
    JobStatus,
    LANGUAGE_MUTABLE_STATUSES,
    NotFound,
    StaleVersion,
)
from ..domain.models import (
    InspectionOutbox,
    InspectionSpec,
    Job,
    MediaInspectionResult,
    MediaType,
    UploadSession,
    now_ms,
)
from ..ids import new_id
from ..ports import MediaInspector, ObjectStorage
from .uow import JobFundingUnitOfWork

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}


def media_type_for(filename: str) -> MediaType:
    return MediaType.AUDIO if Path(filename).suffix.lower() in AUDIO_EXTENSIONS else MediaType.VIDEO


@dataclass(frozen=True)
class UploadTicket:
    upload_id: str
    reserved_job_id: str
    upload_url: str
    object_key: str
    expires_at: int


class JobService:
    def __init__(
        self,
        store: Store,
        storage: ObjectStorage,
        funding: JobFundingUnitOfWork,
        settings: Settings,
        inspector: MediaInspector | None = None,
    ):
        self._store = store
        self._storage = storage
        self._funding = funding
        self._settings = settings
        self._inspector = inspector

    # ------------------------------------------------------------------
    # uploads (§12.2, §12.2.1)
    # ------------------------------------------------------------------

    def create_upload(
        self,
        user_id: str,
        *,
        filename: str,
        size_bytes: int,
        target_language: str,
        file_fingerprint: str = "",
        terms_version: str = "",
        now: int = 0,
    ) -> UploadTicket:
        session = self.prepare_upload(user_id, filename=filename, size_bytes=size_bytes,
            target_language=target_language, file_fingerprint=file_fingerprint, now=now)
        with self._store.transaction() as tx:
            tx.insert(session, session.upload_id)
        return self.upload_ticket(session)

    def prepare_upload(self, user_id: str, *, filename: str, size_bytes: int,
                       target_language: str, file_fingerprint: str = "", now: int = 0) -> UploadSession:
        """Build a reservation so a caller can atomically persist it with its draft."""
        now = now or now_ms()
        if size_bytes <= 0 or size_bytes > self._settings.max_upload_bytes:
            raise DomainError("invalid file size", code=ErrorCode.MEDIA_UNSUPPORTED)
        upload_id = new_id("upl")
        job_id = new_id("job")
        suffix = Path(filename).suffix.lower() or ".bin"
        object_key = f"users/{user_id}/jobs/{job_id}/input{suffix}"
        if self._settings.profile == 'azure-jp-t4':
            object_key = 'inputs/' + object_key
        session = UploadSession(
            upload_id=upload_id,
            owner_user_id=user_id,
            reserved_job_id=job_id,
            object_key=object_key,
            original_filename=Path(filename).name,  # sanitized: never a client path
            declared_size_bytes=size_bytes,
            file_fingerprint=file_fingerprint,
            target_language=target_language,
            sas_expires_at=now + self._settings.sas_ttl_seconds * 1000,
            last_activity_at=now,
            created_at=now,
        )
        return session

    def upload_ticket(self, session: UploadSession) -> UploadTicket:
        url = self._storage.create_upload_url(session.object_key, self._settings.sas_ttl_seconds)
        return UploadTicket(session.upload_id, session.reserved_job_id, url, session.object_key, session.sas_expires_at)

    def renew_upload(self, upload_id: str, user_id: str, *, now: int = 0) -> UploadTicket:
        now = now or now_ms()
        with self._store.transaction() as tx:
            session = self._owned_session(tx, upload_id, user_id)
            if session.status != UploadSession.STATUS_UPLOADING:
                raise InvalidTransition(f"cannot renew upload in status {session.status}")
            session.sas_expires_at = now + self._settings.sas_ttl_seconds * 1000
            session.last_activity_at = now
            tx.put(session, upload_id)
        url = self._storage.create_upload_url(session.object_key, self._settings.sas_ttl_seconds)
        return UploadTicket(upload_id, session.reserved_job_id, url, session.object_key, session.sas_expires_at)

    def get_upload(self, upload_id: str, user_id: str) -> UploadSession:
        with self._store.transaction() as tx:
            return self._owned_session(tx, upload_id, user_id)

    def complete_upload(self, upload_id: str, user_id: str, *, now: int = 0) -> Job:
        """Idempotent commit (§12.2): repeats return the one reserved job."""
        now = now or now_ms()
        with self._store.transaction() as tx:
            session = self._owned_session(tx, upload_id, user_id)
            existing = tx.get(Job, session.reserved_job_id)
            if existing is not None:
                return existing
            if session.status not in (UploadSession.STATUS_UPLOADING, UploadSession.STATUS_COMMITTED):
                raise InvalidTransition(f"cannot complete upload in status {session.status}")
            session.status = UploadSession.STATUS_COMPLETING
            session.last_activity_at = now
            tx.put(session, upload_id)

        if not self._storage.exists(session.object_key, expected_size=session.declared_size_bytes):
            with self._store.transaction() as tx:
                session = self._owned_session(tx, upload_id, user_id)
                session.status = UploadSession.STATUS_UPLOADING
                tx.put(session, upload_id)
            raise DomainError("blob not committed or size mismatch", code=ErrorCode.UPLOAD_INCOMPLETE)

        # Sync probe only when the backend can offer a local path; otherwise
        # the async CPU-inspection outbox path takes over (§12.2).
        result: MediaInspectionResult | None = None
        if self._inspector is not None and session.declared_size_bytes <= self._settings.sync_inspect_max_bytes:
            local = self._storage.local_path(session.object_key)
            if local is not None:
                try:
                    result = self._inspector.inspect(local)
                except DomainError:
                    result = None  # fall through to the async path for a definitive answer

        if result is not None:
            from .uow import CompleteInspectionCommand

            charge = self._funding.complete_inspection(
                CompleteInspectionCommand(
                    job_id=session.reserved_job_id,
                    owner_user_id=user_id,
                    original_filename=session.original_filename,
                    media_type=result.media_type,
                    target_language=session.target_language,
                    input_object_key=session.object_key,
                    duration_ms=result.duration_ms,
                    duration_probe_raw=result.duration_probe_raw,
                    inspection_attempt=1,
                    now=now,
                )
            )
            job = charge.job
        else:
            job = self._create_inspecting_job(session, now)

        with self._store.transaction() as tx:
            session = self._owned_session(tx, upload_id, user_id)
            session.status = UploadSession.STATUS_COMPLETED
            session.completed_at = now
            tx.put(session, upload_id)
        return job

    def _create_inspecting_job(self, session: UploadSession, now: int) -> Job:
        job = Job(
            job_id=session.reserved_job_id,
            owner_user_id=session.owner_user_id,
            original_filename=session.original_filename,
            media_type=media_type_for(session.original_filename),
            target_language=session.target_language,
            status=JobStatus.INSPECTING,
            inspection_attempt=1,
            input_object_key=session.object_key,
            created_at=now,
        )
        spec = InspectionSpec(
            job_id=job.job_id,
            inspection_attempt=1,
            input_uri=f"obj://{session.object_key}",
            result_uri=f"obj://users/{session.owner_user_id}/jobs/{job.job_id}/inspection.json",
        )
        outbox = InspectionOutbox(
            outbox_id=f"inspection:{job.job_id}:1",
            job_id=job.job_id,
            inspection_attempt=1,
            spec=spec.to_json_dict(),
            created_at=now,
        )
        with self._store.transaction() as tx:
            tx.insert(job, job.job_id)
            tx.insert(outbox, outbox.outbox_id)
        return job

    # ------------------------------------------------------------------
    # job operations
    # ------------------------------------------------------------------

    def get_job(self, job_id: str, user_id: str) -> Job:
        with self._store.transaction() as tx:
            return self._owned_job(tx, job_id, user_id)

    def list_jobs(self, user_id: str) -> list[Job]:
        with self._store.transaction() as tx:
            return tx.query(Job, where=("owner_user_id", "==", user_id), order_by="created_at")

    def start(self, job_id: str, user_id: str, *, now: int = 0):
        with self._store.transaction() as tx:
            self._owned_job(tx, job_id, user_id)
        return self._funding.fund_and_enqueue(job_id, now=now)

    def cancel(self, job_id: str, user_id: str, *, now: int = 0):
        with self._store.transaction() as tx:
            self._owned_job(tx, job_id, user_id)
        return self._funding.cancel(job_id, now=now)

    def retry(self, source_job_id: str, user_id: str, *, now: int = 0) -> Job:
        """Create a NEW job referencing the retained input (§3.5, §12.2).

        Idempotent on ``job-retry:<source>:<next_attempt>`` — repeated clicks
        return the same child job instead of charging twice.
        """
        now = now or now_ms()
        with self._store.transaction() as tx:
            source = self._owned_job(tx, source_job_id, user_id)
            self._funding.require_processing()
            recovered_preview = source.error_code == ErrorCode.PROCESSING_UNAVAILABLE.value
            if source.status != JobStatus.FAILED or not (source.retry_allowed or recovered_preview):
                raise DomainError("job is not retryable", code=ErrorCode.RETRY_NOT_ALLOWED)
            if source.retry_expires_at is not None and now > source.retry_expires_at:
                raise DomainError("retry window expired", code=ErrorCode.RETRY_NOT_ALLOWED)
            if source.attempt_number >= self._settings.max_retry_attempts:
                raise DomainError("retry attempts exhausted", code=ErrorCode.RETRY_EXHAUSTED)
        if not self._storage.exists(source.input_object_key):
            raise DomainError("input object no longer retained", code=ErrorCode.RETRY_NOT_ALLOWED)

        next_attempt = source.attempt_number + 1
        idem = f"job-retry:{source_job_id}:{next_attempt}"
        new_job_id = "job_" + hashlib.sha256(idem.encode()).hexdigest()[:16]
        with self._store.transaction() as tx:
            existing = tx.get(Job, new_job_id)
            if existing is not None:
                return existing
            child = Job(
                job_id=new_job_id,
                owner_user_id=user_id,
                original_filename=source.original_filename,
                media_type=source.media_type,
                duration_ms=source.duration_ms,
                duration_probe_raw=source.duration_probe_raw,
                target_language=source.target_language,
                status=JobStatus.AWAITING_CREDITS,  # charge path decides the real state
                attempt_number=next_attempt,
                retry_of_job_id=source.job_id,
                input_owner_job_id=source.input_owner_job_id or source.job_id,
                input_object_key=source.input_object_key,
                quoted_point_units=source.quoted_point_units,
                pricing_version=source.pricing_version,
                terms_version=source.terms_version,
                media_rights_attested_at=source.media_rights_attested_at,
                created_at=now,
            )
            tx.insert(child, new_job_id)
        return self._funding.fund_and_enqueue(new_job_id, now=now).job

    def patch_language(self, job_id: str, user_id: str, *, language: str, expected_status_version: int) -> Job:
        """Optimistic-concurrency language edit (§12.2): one writer wins."""
        with self._store.transaction() as tx:
            job = self._owned_job(tx, job_id, user_id)
            if job.status not in LANGUAGE_MUTABLE_STATUSES:
                raise InvalidTransition(f"language frozen in status {job.status}")
            if job.status_version != expected_status_version:
                raise StaleVersion("job changed; reload and retry")
            job.target_language = language
            tx.put(job, job_id)
            return tx.get(Job, job_id)

    def download_url(self, job_id: str, user_id: str, *, now: int = 0) -> str:
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = self._owned_job(tx, job_id, user_id)
        if job.status != JobStatus.SUCCEEDED or not job.output_object_key:
            raise InvalidTransition("result not available")
        if job.output_expires_at is not None and now > job.output_expires_at:
            raise DomainError("result expired", code=ErrorCode.NOT_FOUND)
        return self._storage.create_download_url(job.output_object_key, self._settings.sas_ttl_seconds)

    def recover_simulated_results(self) -> int:
        """Repair old local-preview successes only when provenance AND bytes match."""
        if self._settings.profile not in {"local-ui", "local-full"}:
            return 0
        with self._store.transaction() as tx:
            jobs = tx.query(Job, where=("status", "==", JobStatus.SUCCEEDED))
        repaired = 0
        for job in jobs:
            if not (job.backend_job_id or "").startswith("fake-") or not job.output_object_key:
                continue
            path = self._storage.local_path(job.output_object_key)
            if path and path.is_file() and path.stat().st_size == 11 and path.read_bytes() == b"fake-result":
                self._funding.invalidate_simulated_result(job.job_id, output_key=job.output_object_key)
                repaired += 1
        return repaired

    def delete_assets(self, job_id: str, user_id: str, *, now: int = 0) -> Job:
        now = now or now_ms()
        with self._store.transaction() as tx:
            job = self._owned_job(tx, job_id, user_id)
            if job.status in ACTIVE_CHARGED_STATUSES or job.status == JobStatus.AWAITING_CAPACITY:
                raise InvalidTransition("cannot delete assets of an active job")
            references = tx.query(Job, where=("input_object_key", "==", job.input_object_key))
            live_refs = [j for j in references if j.job_id != job_id and j.assets_deleted_at is None]
            job.assets_deleted_at = now
            job.asset_cleanup_status = "deleted"
            tx.put(job, job_id)
        if job.output_object_key:
            self._storage.delete(job.output_object_key)
            self._storage.delete(job.output_object_key + '.vtt')
        if not live_refs:
            self._storage.delete(job.input_object_key)
        return job

    # ------------------------------------------------------------------

    def _owned_session(self, tx, upload_id: str, user_id: str) -> UploadSession:
        session = tx.get(UploadSession, upload_id)
        if session is None:
            raise NotFound(f"upload {upload_id}")
        if session.owner_user_id != user_id:
            raise Forbidden("not your upload")
        return session

    def _owned_job(self, tx, job_id: str, user_id: str) -> Job:
        job = tx.get(Job, job_id)
        if job is None:
            raise NotFound(f"job {job_id}")
        if job.owner_user_id != user_id:
            raise Forbidden("not your job")
        return job
