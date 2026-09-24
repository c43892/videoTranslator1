"""Worker-only API. Workers initiate every connection; no home inbound ports."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..application.downloads import DownloadConflict


class Lease(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    lease_token: str = Field(min_length=1, max_length=100)


class Heartbeat(BaseModel):
    active: list[Lease] = Field(default_factory=list, max_length=2)


def install_download_routes(app, service):
    prefix = "/api/v1/download-workers"

    def worker(authorization: str = Header(default="")):
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        worker_id = service.authenticate(token)
        if not worker_id:
            raise HTTPException(401, detail={"code": "worker_unauthenticated"})
        return worker_id

    def call(fn, *args):
        try:
            return fn(*args)
        except DownloadConflict as exc:
            raise HTTPException(409, detail={"code": str(exc)}) from None
        except ValueError as exc:
            raise HTTPException(422, detail={"code": str(exc)}) from None

    @app.post(prefix + "/heartbeat")
    def heartbeat(body: Heartbeat, worker_id=Depends(worker)):
        return call(service.heartbeat, worker_id, [(t.task_id, t.lease_token) for t in body.active])

    @app.post(prefix + "/claim")
    def claim(worker_id=Depends(worker)):
        return {"task": call(service.claim, worker_id)}

    @app.post(prefix + "/fail")
    def fail(body: Lease, worker_id=Depends(worker)):
        call(service.fail, body.task_id, worker_id, body.lease_token)
        return {"failed": True}

    @app.put(prefix + "/tasks/{task_id}/content")
    async def content(task_id: str, request: Request, worker_id=Depends(worker),
                      x_download_lease: str = Header(default="")):
        await run_in_threadpool(call, service.check_upload, task_id, worker_id, x_download_lease)
        length = request.headers.get("content-length", "")
        if not length.isdigit() or not 0 < int(length) <= service.settings.max_upload_bytes:
            raise HTTPException(413, detail={"code": "invalid_file_size"})
        size = int(length)
        with TemporaryDirectory(prefix="vt-home-upload-") as directory:
            path, count = Path(directory) / "video.mp4", 0
            try:
                async with asyncio.timeout(1800):
                    with path.open("wb") as stream:
                        async for chunk in request.stream():
                            count += len(chunk)
                            if count > size:
                                raise HTTPException(413, detail={"code": "invalid_file_size"})
                            await run_in_threadpool(stream.write, chunk)
            except TimeoutError:
                raise HTTPException(408, detail={"code": "upload_timeout"}) from None
            if count != size:
                raise HTTPException(422, detail={"code": "invalid_file_size"})
            return await run_in_threadpool(call, service.complete, task_id, worker_id, x_download_lease, path)
