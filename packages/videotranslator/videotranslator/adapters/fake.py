"""In-process fake adapters: the ``test`` profile and unit tests (§13.1)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..domain.enums import BackendError, FailureClass
from ..domain.models import (
    BackendJobRef,
    BackendStatus,
    InspectionSpec,
    JobSpec,
    MediaInspectionResult,
    TimedSegment,
)
from ..ports import (
    PaymentEventData,
    PaymentSession,
    PaymentSessionRequest,
    PaymentSnapshot,
    UserIdentity,
)


class FakeIdentityVerifier:
    """Tokens look like ``fake:<uid>[:unverified]`` — deterministic for tests."""

    def verify(self, bearer_token: str) -> UserIdentity:
        parts = bearer_token.split(":")
        if len(parts) < 2 or parts[0] != "fake":
            raise BackendError("invalid token", failure_class=FailureClass.PERMANENT)
        uid = parts[1]
        verified = "unverified" not in parts[2:]
        return UserIdentity(uid=uid, email=f"{uid}@example.test", email_verified=verified, display_name=uid)


class FakeObjectStorage:
    """Directory-backed storage so worker/backend flows touch real files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in Path(key).parts:
            raise BackendError("unsafe object key", failure_class=FailureClass.PERMANENT)
        return self.root / key

    def create_upload_url(self, object_key: str, expires_in: int) -> str:
        return f"fake://upload/{object_key}"

    def create_download_url(self, object_key: str, expires_in: int) -> str:
        return f"fake://download/{object_key}"

    def download(self, object_key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(object_key), destination)
        return destination

    def upload(self, source: Path, object_key: str) -> str:
        target = self._path(object_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return object_key

    def put_bytes(self, object_key: str, data: bytes) -> None:
        target = self._path(object_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def exists(self, object_key: str, *, expected_size: int | None = None) -> bool:
        path = self._path(object_key)
        if not path.exists():
            return False
        return expected_size is None or path.stat().st_size == expected_size

    def local_path(self, object_key: str) -> Path | None:
        return self._path(object_key)

    def delete(self, object_key: str) -> None:
        self._path(object_key).unlink(missing_ok=True)


class FakeJobBackend:
    """Scripted backend: each ``get_status`` poll advances one state."""

    name = "fake"

    def __init__(self, storage: FakeObjectStorage | None = None, *, fail_on_poll: int | None = None):
        self._storage = storage
        self._jobs: dict[str, dict] = {}
        self._idem: dict[str, str] = {}
        self._fail_on_poll = fail_on_poll
        self.submitted_specs: list[JobSpec] = []

    def submit(self, spec: JobSpec, idempotency_key: str) -> BackendJobRef:
        if idempotency_key in self._idem:
            return BackendJobRef(self._idem[idempotency_key], idempotency_key)
        backend_id = f"fake-{len(self._jobs)}"
        self._jobs[backend_id] = {"spec": spec, "polls": 0, "cancelled": False}
        self._idem[idempotency_key] = backend_id
        self.submitted_specs.append(spec)
        return BackendJobRef(backend_id, idempotency_key)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        record = self._jobs.get(backend_job_id)
        if record is None:  # unknown-outcome probe by deterministic name
            mapped = self._idem.get(backend_job_id)
            record = self._jobs.get(mapped or "", None)
            if record is None:
                return BackendStatus(state="not_found")
        if record["cancelled"]:
            return BackendStatus(state="cancelled", started_at=record.get("started_at"))
        record["polls"] += 1
        polls = record["polls"]
        if self._fail_on_poll is not None and polls >= self._fail_on_poll:
            return BackendStatus(state="failed", error_message="scripted failure", actual_gpu_seconds=polls)
        if polls == 1:
            return BackendStatus(state="provisioning")
        if polls == 2:
            record["started_at"] = 1
            return BackendStatus(state="running", stage="transcribe", progress_percent=40, started_at=1)
        spec: JobSpec = record["spec"]
        output_key = spec.output_uri.removeprefix("obj://")
        if self._storage is not None:
            self._storage.put_bytes(output_key, b"fake-result")
        return BackendStatus(
            state="succeeded", progress_percent=100, output_object_key=output_key, actual_gpu_seconds=3
        )

    def cancel(self, backend_job_id: str) -> bool:
        record = self._jobs.get(backend_job_id)
        if record is None:
            return False
        record["cancelled"] = True
        return True


class FakeMediaInspectionBackend:
    """Returns a canned result after one running poll; scriptable failure."""

    name = "fake-inspection"

    def __init__(self, result: MediaInspectionResult | None = None, *, fail: bool = False):
        self._result = result or MediaInspectionResult(
            duration_ms=60_000, duration_probe_raw="60.0", media_type="video", has_audio=True
        )
        self._fail = fail
        self._jobs: dict[str, int] = {}

    def submit(self, spec: InspectionSpec, idempotency_key: str) -> BackendJobRef:
        self._jobs.setdefault(idempotency_key, 0)
        return BackendJobRef(idempotency_key, idempotency_key)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        if self._fail:
            return BackendStatus(state="failed", error_message="scripted inspection failure")
        polls = self._jobs.get(backend_job_id, 0) + 1
        self._jobs[backend_job_id] = polls
        return BackendStatus(state="running") if polls < 2 else BackendStatus(state="succeeded")

    def get_result(self, backend_job_id: str) -> MediaInspectionResult:
        return self._result


class FakePaymentGateway:
    """Produces signed-looking events without any network; drives §10 tests."""

    def __init__(self, provider: str):
        self.provider = provider
        self._sessions: dict[str, PaymentSessionRequest] = {}

    def create_session(self, request: PaymentSessionRequest) -> PaymentSession:
        self._sessions[request.payment_id] = request
        order_id = f"order-{request.payment_id}" if self.provider == "paypal" else None
        return PaymentSession(
            payment_id=request.payment_id,
            provider=self.provider,
            confirmation_mode="server_capture" if self.provider == "paypal" else "webhook",
            redirect_url=f"fake://pay/{self.provider}/{request.payment_id}",
            provider_order_id=order_id,
        )

    def capture(self, provider_order_id: str, idempotency_key: str) -> PaymentSnapshot:
        payment_id = provider_order_id.removeprefix("order-")
        request = self._sessions[payment_id]
        return PaymentSnapshot(
            provider_payment_id=f"pp-{payment_id}",
            status="completed",
            amount_minor=request.package.amount_minor,
            currency=request.package.currency,
            user_id=request.user_id,
            provider_capture_id=f"cap-{payment_id}",
        )

    def make_webhook(self, payment_id: str) -> tuple[dict, bytes]:
        """Test helper: a verified checkout.completed event for a session."""
        request = self._sessions[payment_id]
        body = json.dumps(
            {"id": f"evt_{payment_id}", "payment_id": payment_id, "amount": request.package.amount_minor}
        ).encode()
        return {}, body

    def verify_webhook(self, headers, raw_body: bytes) -> PaymentEventData:
        data = json.loads(raw_body)
        payment_id = data["payment_id"]
        request = self._sessions[payment_id]
        return PaymentEventData(
            provider=self.provider,
            event_id=data["id"],
            event_type="checkout.session.completed",
            payment_id=payment_id,
            provider_payment_id=f"{self.provider[:2]}-{payment_id}",
            provider_capture_id=None,
            amount_minor=request.package.amount_minor,
            currency=request.package.currency,
            payload={},
        )

    def get_payment(self, provider_payment_id: str) -> PaymentSnapshot:
        payment_id = provider_payment_id.split("-", 1)[1]
        request = self._sessions[payment_id]
        return PaymentSnapshot(
            provider_payment_id=provider_payment_id,
            status="completed",
            amount_minor=request.package.amount_minor,
            currency=request.package.currency,
        )

    def refund(self, provider_payment_id: str, amount_minor: int) -> bool:
        return True


# ---------------------------------------------------------------------------
# Worker-side fake processors (no GPU, no models)
# ---------------------------------------------------------------------------


class FakeSourceSeparator:
    def separate_vocals(self, audio: Path, out_dir: Path) -> Path:
        out = out_dir / "vocals.wav"
        shutil.copyfile(audio, out)
        return out


class FakeSpeechTranscriber:
    def __init__(self, segments: list[TimedSegment] | None = None):
        self._segments = segments

    def transcribe(self, audio: Path, source_language: str | None) -> list[TimedSegment]:
        if self._segments is not None:
            return self._segments
        sidecar = audio.with_suffix(".segments.json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text())
            return [TimedSegment(**s) for s in data]
        return [TimedSegment(index=0, start_ms=0, end_ms=3000, source_text="hello world")]


class FakeTranslationProvider:
    def __init__(self):
        self.calls: list[dict] = []

    def translate(self, segments, target_language, target_durations_ms=None):
        self.calls.append({"n": len(segments), "target": target_language, "durations": target_durations_ms})
        return [
            TimedSegment(
                index=s.index,
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                source_text=s.source_text,
                translated_text=f"[{target_language}] {s.source_text}",
            )
            for s in segments
        ]


class FakeVoiceCloner:
    """Copies the reference clip as the 'generated' speech (deterministic)."""

    def synthesize(self, text: str, reference_audio: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_audio, destination)
        return destination
