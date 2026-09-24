"""Async, cancellable separation API sharing a GPU mutex with IndexTTS2."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
JOBS = {}
JOBS_LOCK = threading.Lock()


class Separation(BaseModel):
    audio: str
    prefix: str
    model: Literal['htdemucs','htdemucs_ft'] = 'htdemucs'
    segment: float = Field(default=5,ge=1,le=7.8)


@contextmanager
def child_process(*args, **kwargs):
    proc = subprocess.Popen(*args, **kwargs)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()


def install(app, gpu_lock, media_path, unload):
    def status(job):
        result = {k:v for k,v in job.items() if k in ('id','status','stems','error')}
        progress = job['directory']/'progress.json'
        if progress.exists():
            try:result.update(json.loads(progress.read_text()))
            except (ValueError,FileNotFoundError):pass
        return result

    def execute(job, body):
        proc = None
        try:
            with gpu_lock:
                if job['cancel'].is_set():
                    job['status']='cancelled'
                    return
                unload()
                directory=job['directory']
                directory.mkdir(parents=True,exist_ok=True)
                config=directory/'request.json'
                config.write_text(json.dumps({'input':str(media_path(body.audio)),
                    'directory':str(directory),'model':body.model,'segment':body.segment}))
                job['status']='running'
                env={k:v for k,v in os.environ.items() if k in ('PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES')}
                env.update(TORCH_HOME='/models/demucs',PYTHONUNBUFFERED='1',HOME='/models')
                with (directory/'process.log').open('w') as log:
                    with child_process([sys.executable,str(Path(__file__).with_name('demucs_worker.py')),str(config)],
                        stdout=log,stderr=log,env=env,start_new_session=True) as proc:
                        deadline=time.monotonic()+24*3600
                        while proc.poll() is None:
                            if job['cancel'].is_set() or time.monotonic()>deadline:
                                os.killpg(proc.pid,signal.SIGTERM)
                                try:proc.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                                job['status']='cancelled' if job['cancel'].is_set() else 'failed'
                                job['error']='Demucs 分离已取消或超时'
                                return
                            time.sleep(.25)
                        if proc.returncode != 0:
                            raise RuntimeError('Demucs 分离失败，请检查 GPU 服务日志后重试')
                job['stems']={'dialogue':body.prefix+'/dialogue.wav','music':body.prefix+'/background.wav',
                              'effects':body.prefix+'/effects.wav'}
                job['status']='completed'
        except Exception as exc:
            job.update(status='failed',error=str(exc)[:500])
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            job['finished'].set()

    @router.post('/separations',status_code=202)
    def create(body: Separation):
        source,directory=media_path(body.audio),media_path(body.prefix)
        if not source.is_file():raise HTTPException(422,'分离输入音频不存在')
        job_id=hashlib.sha256(body.model_dump_json().encode()).hexdigest()[:24]
        with JOBS_LOCK:
            prior=JOBS.get(job_id)
            if prior and prior['status'] in ('queued','running','completed'):return status(prior)
            if any(j['status'] in ('queued','running') for j in JOBS.values()):
                raise HTTPException(409,'GPU 分离队列忙，请稍后重试')
            # Bound in-memory history. Completed files remain in the job's storage.
            for key in list(JOBS):
                if len(JOBS)>100 and JOBS[key]['status'] not in ('queued','running'):del JOBS[key]
            job={'id':job_id,'status':'queued','directory':directory,'cancel':threading.Event(),'finished':threading.Event()}
            JOBS[job_id]=job
            threading.Thread(target=execute,args=(job,body),daemon=True).start()
            return status(job)

    @router.get('/separations/{job_id}')
    def get(job_id:str):
        with JOBS_LOCK:
            if job_id not in JOBS:raise HTTPException(404,'Separation not found')
            return status(JOBS[job_id])

    @router.delete('/separations/{job_id}')
    def cancel(job_id:str):
        with JOBS_LOCK:
            if job_id not in JOBS:raise HTTPException(404,'Separation not found')
            job=JOBS[job_id]
            job['cancel'].set()
        if not job['finished'].wait(15):raise HTTPException(503,'Demucs cancellation is pending')
        return status(job)

    app.include_router(router)
