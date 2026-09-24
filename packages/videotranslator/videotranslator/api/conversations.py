"""Chat HTTP boundary and local storage bridge; auth remains the existing identity port."""
import asyncio
import contextlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..application.pricing import public_price
from ..domain.enums import DomainError
from ..domain.models import UploadSession


class NewConversation(BaseModel):
    locale: str = Field(default="en", max_length=35)


class EditConversation(BaseModel):
    revision: int = Field(ge=0)
    text: str = Field(default="", max_length=2000)
    choice: Literal["", "source", "target", "locale", "edit_source", "edit_target"] = ""
    value: str = Field(default="", max_length=35)
    filename: str = Field(default="", max_length=255)
    size_bytes: int = Field(default=0, ge=0)


class ConfirmConversation(BaseModel):
    revision: int = Field(ge=0)


class FollowRetry(BaseModel):
    job_id: str = Field(max_length=80)


def install_conversation_routes(app, container, identity_dependency, http_error):
    service = container.conversations
    if service.home_downloads:
        from .downloads import install_download_routes
        install_download_routes(app, service.home_downloads)
    local = container.settings.profile in {"test", "local-ui", "local-full"}

    def invoke(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except DomainError as exc:
            raise http_error(exc) from exc
        except ValueError as exc:
            raise HTTPException(422, detail={"code": str(exc)}) from exc

    @app.get("/api/v1/chat/config")
    def config():
        return {"local": local, "profile": container.settings.profile,
                "processing_available": container.settings.processing_available,
                "demo": container.settings.profile == "test" or not container.settings.processing_available,
                "auth_mode": "demo" if container.settings.profile == "test" else container.settings.auth_mode,
                "firebase": container.settings.firebase_web_config,
                "payment_mode": container.settings.payment_mode,
                "payment_providers": list(container.gateways) if container.settings.auth_mode != "demo" and container.settings.profile != "test" else [],
                **public_price(container.funding.pricing.current()),
                "max_upload_bytes": container.settings.max_upload_bytes,
                "target_languages": ["zh", "en"]}

    @app.post("/api/v1/conversations", status_code=201)
    def create(body: NewConversation, identity=Depends(identity_dependency)):
        return service.view(invoke(service.create, identity.uid, body.locale))

    @app.get("/api/v1/conversations/{key}")
    def get(key: str, identity=Depends(identity_dependency)):
        return service.view(invoke(service.get, key, identity.uid))

    @app.post("/api/v1/conversations/{key}/messages")
    def edit(key: str, body: EditConversation, identity=Depends(identity_dependency)):
        return service.view(invoke(service.edit, key, identity.uid, **body.model_dump()))

    @app.post("/api/v1/conversations/{key}/confirm")
    def confirm(key: str, body: ConfirmConversation, identity=Depends(identity_dependency)):
        return service.view(invoke(service.confirm, key, identity.uid, body.revision))

    @app.post("/api/v1/conversations/{key}/prepare")
    def prepare(key: str, body: ConfirmConversation, identity=Depends(identity_dependency)):
        return service.view(invoke(service.prepare, key, identity.uid, body.revision))

    @app.post("/api/v1/conversations/{key}/continue")
    def continue_conversation(key: str, body: ConfirmConversation, identity=Depends(identity_dependency)):
        return service.view(invoke(service.continue_conversation, key, identity.uid, body.revision))

    @app.post("/api/v1/conversations/{key}/inspect")
    def inspect(key: str, identity=Depends(identity_dependency)):
        return service.view(invoke(service.finish_preparation, key, identity.uid))

    @app.post("/api/v1/conversations/{key}/follow-retry")
    def follow_retry(key: str, body: FollowRetry, identity=Depends(identity_dependency)):
        return service.view(invoke(service.follow_retry, key, identity.uid, body.job_id))

    @app.put("/api/v1/uploads/{upload_id}/content")
    async def upload_content(upload_id: str, request: Request, identity=Depends(identity_dependency)):
        if not local:
            raise HTTPException(404)
        session = invoke(container.jobs.get_upload, upload_id, identity.uid)
        if session.status != UploadSession.STATUS_UPLOADING:
            raise HTTPException(409, detail={"code": "already_uploaded"})
        count = 0
        # A unique staging path prevents concurrent retries from corrupting a partial upload.
        with TemporaryDirectory(prefix="vt-upload-") as directory:
            path = Path(directory) / "input"
            with path.open("wb") as stream:
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > session.declared_size_bytes:
                        raise HTTPException(413, detail={"code": "invalid_file_size"})
                    await run_in_threadpool(stream.write, chunk)
            if count != session.declared_size_bytes:
                raise HTTPException(422, detail={"code": "invalid_file_size"})
            await run_in_threadpool(container.storage.upload, path, session.object_key)
        return {"uploaded": True}

    @app.get("/api/v1/jobs/{job_id}/content")
    def result_content(job_id: str, identity=Depends(identity_dependency)):
        if not local:
            raise HTTPException(404)
        invoke(container.jobs.download_url, job_id, identity.uid)
        job = invoke(container.jobs.get_job, job_id, identity.uid)
        path = container.storage.local_path(job.output_object_key)
        if path is None or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename="translated.mp3" if job.media_type == "audio" else "translated.mp4")

    @app.get('/api/v1/jobs/{job_id}/playback')
    def playback(job_id: str, expires: int, token: str, download: bool = False, asset: str = 'media'):
        # A short-lived capability minted only by the authenticated result route.
        # Native media elements can then seek using standard HTTP Range requests.
        from ..domain.models import Job, now_ms
        from ..domain.enums import JobStatus
        if not local or not hasattr(container.storage, 'verify_url'):
            raise HTTPException(404)
        with container.store.transaction() as tx:
            job = tx.get(Job, job_id)
        if (not job or job.status != JobStatus.SUCCEEDED or not job.output_object_key
                or job.assets_deleted_at or (job.output_expires_at and job.output_expires_at < now_ms())):
            raise HTTPException(404)
        if asset not in ('media', 'subtitles') or (asset == 'subtitles' and job.media_type != 'video'):
            raise HTTPException(404)
        key = job.output_object_key + ('.vtt' if asset == 'subtitles' else '')
        if not container.storage.verify_url(key, expires, token):
            raise HTTPException(403)
        path = container.storage.local_path(key)
        if not path or not path.is_file():
            raise HTTPException(404)
        if asset == 'subtitles':
            return FileResponse(path, media_type='text/vtt',
                headers={'Cache-Control': 'private, no-store', 'Referrer-Policy': 'no-referrer'})
        return FileResponse(path, media_type='audio/mpeg' if job.media_type == 'audio' else 'video/mp4',
            filename='translated.mp3' if job.media_type == 'audio' else 'translated.mp4',
            content_disposition_type='attachment' if download else 'inline',
            headers={'Cache-Control':'private, no-store', 'Referrer-Policy':'no-referrer'})

    @app.on_event("startup")
    async def start_importer():
        async def heartbeat_monitor():
            while True:
                if service.home_downloads and service.home_downloads.enabled:
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(service.home_downloads.tick)
                await asyncio.sleep(1)
        async def loop():
            while True:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(service.import_one)
                await asyncio.sleep(2)
        app.state.import_task = asyncio.create_task(loop())
        app.state.download_monitor_task = asyncio.create_task(heartbeat_monitor())

    @app.on_event("shutdown")
    async def stop_tasks():
        for name in ("import_task", "processor_task", "download_monitor_task"):
            task = getattr(app.state, name, None)
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    static = Path(__file__).resolve().parent.parent / "web"
    app.mount("/assets", StaticFiles(directory=static), name="chat-assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})
