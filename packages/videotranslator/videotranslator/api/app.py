"""Control API (§12): thin HTTP layer over the application services.

Routes translate HTTP ↔ domain; every business rule lives in the
application layer. Background processors (dispatcher / reconciler / inbox)
run as an in-process loop for local profiles and as Container Apps Jobs in
Azure — same ``run_once`` entry points either way.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field

from ..bootstrap import Container
from ..domain.enums import (
    BackendError,
    DomainError,
    EmailNotVerified,
    ErrorCode,
    Forbidden,
    FailureClass,
    InsufficientCredits,
    InvalidTransition,
    NotFound,
    StaleVersion,
)
from ..domain.models import Job, User, now_ms
from ..ports import UserIdentity

_ERROR_STATUS = {
    ErrorCode.PROCESSING_UNAVAILABLE: 503,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.EMAIL_NOT_VERIFIED: 403,
    ErrorCode.INSUFFICIENT_CREDITS: 402,
    ErrorCode.STALE_JOB_VERSION: 409,
    ErrorCode.INVALID_TRANSITION: 409,
    ErrorCode.UPLOAD_INCOMPLETE: 409,
    ErrorCode.CAPACITY_UNAVAILABLE: 409,
    ErrorCode.RETRY_NOT_ALLOWED: 409,
    ErrorCode.RETRY_EXHAUSTED: 409,
    ErrorCode.MEDIA_TOO_LONG: 422,
    ErrorCode.MEDIA_UNSUPPORTED: 422,
}


def _http_error(exc: DomainError) -> HTTPException:
    return HTTPException(_ERROR_STATUS.get(exc.code, 400), detail={"code": exc.code.value, "message": str(exc)})


class CreateUploadRequest(BaseModel):
    filename: str
    size_bytes: int = Field(gt=0)
    target_language: str
    file_fingerprint: str = ""


class CreateSessionRequest(BaseModel):
    package_id: str
    provider: str
    success_url: str = ""
    cancel_url: str = ""
    idempotency_key: str = Field(default="", max_length=100)


class UpdatePricingRequest(BaseModel):
    expected_version: str = Field(min_length=1, max_length=100)
    rate_cents_per_minute: int = Field(strict=True, ge=1, le=10000)
    minimum_cents: int = Field(strict=True, ge=1, le=10000)
    billing_increment_cents: int = Field(strict=True, ge=1, le=10000)


class PatchJobRequest(BaseModel):
    target_language: str
    expected_status_version: int


def create_app(container: Container) -> FastAPI:
    app = FastAPI(title="VideoTranslator 2.0 Control API", version="2.0.0")

    @app.middleware('http')
    async def check_browser_payment_mode(request: Request, call_next):
        expected = request.headers.get('x-payment-mode')
        if expected and expected != container.settings.payment_mode and request.url.path.startswith('/api/v1/'):
            return JSONResponse(status_code=409, content={'detail': {'code': 'payment_environment_changed'}})
        return await call_next(request)

    @app.exception_handler(BackendError)
    @app.exception_handler(httpx.RequestError)
    async def backend_unavailable(request, exc):
        return JSONResponse(status_code=503, content={"detail": {"code": "backend_failed"}})

    # -- auth ---------------------------------------------------------------

    def current_identity(authorization: Annotated[str | None, Header()] = None) -> UserIdentity:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, detail={"code": "unauthenticated"})
        try:
            identity = container.identity.verify(authorization.removeprefix("Bearer ").strip())
        except BackendError as exc:
            if exc.failure_class != FailureClass.PERMANENT:
                raise HTTPException(503, detail={"code": "auth_unavailable"}, headers={"Retry-After": "1"}) from None
            raise HTTPException(401, detail={"code": "unauthenticated"}) from None
        _ensure_user(container, identity)
        return identity

    def verified_identity(identity: UserIdentity = Depends(current_identity)) -> UserIdentity:
        if not identity.email_verified:
            raise _http_error(EmailNotVerified("email verification required"))
        return identity

    def admin_identity(identity=Depends(verified_identity)):
        if identity.uid not in container.settings.admin_user_ids:
            raise HTTPException(403, detail={"code": "forbidden"})
        return identity

    @app.get("/api/v1/admin/pricing")
    def admin_pricing(identity=Depends(admin_identity)):
        from ..application.pricing import public_price
        return public_price(container.funding.pricing.current())

    @app.put("/api/v1/admin/pricing")
    def update_pricing(body: UpdatePricingRequest, identity=Depends(admin_identity)):
        from ..application.pricing import public_price
        try:
            return public_price(container.funding.pricing.update(expected_version=body.expected_version,
                rate=body.rate_cents_per_minute, minimum=body.minimum_cents,
                increment=body.billing_increment_cents, actor=identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    # -- helpers --------------------------------------------------------------

    def job_view(job: Job) -> dict:
        data = asdict(job)
        data["status"] = str(job.status)
        return data

    # -- §12.1 identity -------------------------------------------------------

    @app.get("/api/v1/me")
    def me(identity: UserIdentity = Depends(current_identity)):
        with container.store.transaction() as tx:
            user = tx.get(User, identity.uid)
        return {"user_id": identity.uid, "email": identity.email,
                "email_verified": identity.email_verified,
                "is_admin": identity.email_verified and identity.uid in container.settings.admin_user_ids,
                "point_balance_units": user.point_balance_units if user else 0,
                "balance_cents": user.point_balance_units if user else 0, "currency": "USD",
                "billing_status": user.billing_status if user else "clear"}

    @app.get("/api/v1/me/ledger")
    def my_ledger(identity: UserIdentity = Depends(current_identity)):
        from ..domain.models import LedgerEntry

        with container.store.transaction() as tx:
            entries = tx.query(LedgerEntry, where=("user_id", "==", identity.uid), order_by="created_at")
        return {"entries": [asdict(e) for e in entries]}

    # -- §12.2 uploads & jobs ---------------------------------------------------

    @app.post("/api/v1/uploads", status_code=201)
    def create_upload(body: CreateUploadRequest, identity: UserIdentity = Depends(verified_identity)):
        try:
            return asdict(container.jobs.create_upload(
                identity.uid,
                filename=body.filename,
                size_bytes=body.size_bytes,
                target_language=body.target_language,
                file_fingerprint=body.file_fingerprint,
            ))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/v1/uploads/{upload_id}")
    def get_upload(upload_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            return asdict(container.jobs.get_upload(upload_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/uploads/{upload_id}/renew")
    def renew_upload(upload_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            return asdict(container.jobs.renew_upload(upload_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/uploads/{upload_id}/complete", status_code=201)
    def complete_upload(upload_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            from ..domain.conversation import Conversation
            with container.store.transaction() as tx:
                drafts = tx.query(Conversation, where=("upload_id", "==", upload_id))
            if drafts:
                raise InvalidTransition("use_conversation_confirmation")
            return job_view(container.jobs.complete_upload(upload_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/v1/jobs")
    def list_jobs(identity: UserIdentity = Depends(current_identity)):
        return {"jobs": [job_view(j) for j in container.jobs.list_jobs(identity.uid)]}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, identity: UserIdentity = Depends(current_identity)):
        try:
            return job_view(container.jobs.get_job(job_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.patch("/api/v1/jobs/{job_id}")
    def patch_job(job_id: str, body: PatchJobRequest, identity: UserIdentity = Depends(verified_identity)):
        try:
            return job_view(container.jobs.patch_language(
                job_id, identity.uid,
                language=body.target_language,
                expected_status_version=body.expected_status_version,
            ))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/jobs/{job_id}/start")
    def start_job(job_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            result = container.jobs.start(job_id, identity.uid)
            return {"outcome": result.outcome, "job": job_view(result.job)}
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/jobs/{job_id}/retry", status_code=201)
    def retry_job(job_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            return job_view(container.jobs.retry(job_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            result = container.jobs.cancel(job_id, identity.uid)
            return {"outcome": result.outcome, "job": job_view(result.job)}
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/v1/jobs/{job_id}/result")
    def job_result(job_id: str, identity: UserIdentity = Depends(current_identity)):
        try:
            url = container.jobs.download_url(job_id, identity.uid)
            if url.startswith('local://download/'):
                from urllib.parse import urlsplit
                url = f'/api/v1/jobs/{job_id}/playback?' + urlsplit(url).query
            result = {"download_url": url}
            job = container.jobs.get_job(job_id, identity.uid)
            subtitle_key = job.output_object_key + '.vtt'
            if job.media_type == 'video' and container.storage.exists(subtitle_key):
                subtitle_url = container.storage.create_download_url(subtitle_key, container.settings.sas_ttl_seconds)
                if subtitle_url.startswith('local://download/'):
                    from urllib.parse import urlsplit
                    subtitle_url = f'/api/v1/jobs/{job_id}/playback?asset=subtitles&' + urlsplit(subtitle_url).query
                result['subtitle_url'] = subtitle_url
                result['subtitle_language'] = job.target_language
            return result
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.delete("/api/v1/jobs/{job_id}/assets")
    def delete_assets(job_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            return job_view(container.jobs.delete_assets(job_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    # -- §12.3 billing ------------------------------------------------------------

    @app.get("/api/v1/billing/packages")
    def packages(currency: str = "USD", identity: UserIdentity = Depends(current_identity)):
        return {"packages": [asdict(p) for p in container.billing.list_packages(currency)]}

    @app.post("/api/v1/billing/sessions", status_code=201)
    def create_session(body: CreateSessionRequest, identity: UserIdentity = Depends(verified_identity)):
        try:
            return asdict(container.billing.create_session(
                identity.uid,
                package_id=body.package_id,
                provider=body.provider,
                success_url=container.settings.public_app_url + "/?checkout=return",
                cancel_url=container.settings.public_app_url + "/?checkout=cancel",
                idempotency_key=body.idempotency_key,
            ))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/billing/paypal/orders/{order_id}/capture")
    def capture_paypal(order_id: str, identity: UserIdentity = Depends(verified_identity)):
        try:
            key = container.billing.capture_paypal(order_id, identity.uid)
            container.inbox.process_due()  # best-effort inline processing
            return {"inbox_key": key}
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.get("/api/v1/billing/payments/{payment_id}")
    def get_payment(payment_id: str, identity: UserIdentity = Depends(current_identity)):
        try:
            return asdict(container.billing.get_payment(payment_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post('/api/v1/billing/payments/{payment_id}/reconcile')
    def reconcile_payment(payment_id: str, identity: UserIdentity = Depends(current_identity)):
        try:
            key = container.billing.reconcile_payment(payment_id, identity.uid)
            if key:
                container.billing_uow.apply_verified_payment(key)
            return asdict(container.billing.get_payment(payment_id, identity.uid))
        except DomainError as exc:
            raise _http_error(exc) from exc

    @app.post("/api/v1/webhooks/stripe")
    async def stripe_webhook(request: Request):
        return await _webhook(request, "stripe")

    @app.post("/api/v1/webhooks/paypal")
    async def paypal_webhook(request: Request):
        return await _webhook(request, "paypal")

    async def _webhook(request: Request, provider: str):
        body = await request.body()
        try:
            key = container.billing.record_webhook(provider, dict(request.headers), body)
        except (BackendError, DomainError, ValueError, KeyError) as exc:
            raise HTTPException(400, detail={"code": "invalid_webhook", "message": str(exc)}) from exc
        return {"received": True, "inbox_key": key}  # 2xx fast; inbox processor posts later

    # -- §12.4 ops ------------------------------------------------------------------

    @app.get("/api/v1/health/live")
    def live():
        return {"ok": True}

    @app.get("/api/v1/health/ready")
    def ready():
        with container.store.transaction() as tx:
            tx.query(User, limit=1)
        return {"ok": True}

    # -- local background processors ------------------------------------------------

    @app.on_event("startup")
    async def start_processors():
        if container.settings.profile.startswith("azure") and container.settings.profile != "azure-jp-t4":
            return  # Container Apps Jobs run the processors in cloud profiles
        container.seed()
        container.jobs.recover_simulated_results()

        def tick():
            container.dispatcher.dispatch_due_jobs()
            container.dispatcher.dispatch_due_inspections()
            container.inbox.process_due()
            container.reconciler.reconcile_once()

        async def loop():
            while True:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(tick)
                await asyncio.sleep(1)

        task = asyncio.create_task(loop())
        app.state.processor_task = task

    from .conversations import install_conversation_routes
    install_conversation_routes(app, container, verified_identity, _http_error)
    return app


def _ensure_user(container: Container, identity: UserIdentity) -> None:
    """Auto-provision the business user keyed by Firebase uid (§9.1)."""
    with container.store.transaction() as tx:
        if tx.get(User, identity.uid) is None:
            tx.insert(
                User(
                    user_id=identity.uid,
                    email=identity.email,
                    display_name=identity.display_name,
                    photo_url=identity.photo_url,
                    created_at=now_ms(),
                    updated_at=now_ms(),
                ),
                identity.uid,
            )
