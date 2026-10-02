"""Outbound-only GPU agent. One process owns one local TTS/Demucs service."""
from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx
from runtime_status import write_status

PREFIX = '/api/v1/gpu-workers'
ROOT = Path(os.environ.get('GPU_AGENT_DATA_ROOT', '/data')).resolve()
SERVER = os.environ.get('GPU_AGENT_SERVER_URL', '').rstrip('/')
TOKEN = os.environ.get('GPU_WORKER_TOKEN', '')
TTS = os.environ.get('GPU_AGENT_TTS_URL', 'http://tts:8001').rstrip('/')


def gpu_active() -> bool:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=3, check=True)
        return any(float(line.strip()) > 0 for line in result.stdout.splitlines())
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


class Agent:
    def __init__(self):
        if not SERVER.startswith('https://') or len(TOKEN) < 32:
            raise ValueError('A HTTPS server URL and dedicated 32-character GPU worker token are required')
        self.remote = httpx.Client(base_url=SERVER, headers={'Authorization': 'Bearer ' + TOKEN},
                                   timeout=httpx.Timeout(30, read=120, write=120))
        self.local = httpx.Client(base_url=TTS, timeout=httpx.Timeout(30, read=None, write=120))
        self.active: dict | None = None
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.stopping = threading.Event()
        self.provider_id = os.getenv('GPU_PROVIDER_ID') or 'gpu_' + hashlib.sha256(TOKEN.encode()).hexdigest()[:24]
        self.provider_type = os.getenv('GPU_PROVIDER_TYPE', 'local')

    def request(self, method, path, **kwargs):
        response = self.remote.request(method, PREFIX + path, **kwargs)
        response.raise_for_status()
        return response

    def ready(self):
        try:
            response = self.local.get('/health', timeout=5)
            return response.status_code == 200 and response.json().get('status') == 'ready'
        except (httpx.HTTPError, ValueError):
            return False

    def heartbeat_loop(self):
        while not self.stopping.is_set():
            with self.lock:
                active = self.active
            body = {'ready': self.ready(), 'gpu_active': bool(active) and gpu_active(),
                    'provider_id': self.provider_id, 'provider_type': self.provider_type,
                    'active': [{'task_id': active['id'], 'lease_token': active['lease_token']}] if active else []}
            try:
                response = self.request('POST', '/heartbeat', json=body).json()
                if active and active['id'] in response.get('cancelled', []):
                    self.cancelled.set()
                try:
                    write_status(self.provider_id, ready=body['ready'], registered=True,
                                 acknowledged_at=time.time(), busy=bool(active))
                except OSError as exc:
                    print('GPU agent health status unavailable:', type(exc).__name__, flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                print('GPU heartbeat unavailable:', type(exc).__name__, flush=True)
            self.stopping.wait(5)

    def inputs(self, task, directory):
        keys = {}
        headers = {'X-Gpu-Lease': task['lease_token']}
        for name in task['inputs']:
            destination = directory / (name + '.wav' if name != 'audio' else 'source.flac')
            with self.remote.stream('GET', PREFIX + f"/tasks/{task['id']}/inputs/{name}",
                                    headers=headers) as response:
                response.raise_for_status()
                with destination.open('wb') as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        if self.cancelled.is_set():
                            raise RuntimeError('GPU task lease was cancelled')
                        stream.write(chunk)
            keys[name] = destination.relative_to(ROOT).as_posix()
        return keys

    def upload(self, task, name, path):
        if self.cancelled.is_set():
            raise RuntimeError('GPU task lease was cancelled')
        headers = {'X-Gpu-Lease': task['lease_token'], 'Content-Type': 'audio/wav',
                   'Content-Length': str(path.stat().st_size)}
        with path.open('rb') as stream:
            self.request('PUT', f"/tasks/{task['id']}/outputs/{name}", headers=headers,
                         content=iter(lambda: stream.read(1024 * 1024), b''))

    def run_one(self, task):
        directory = (ROOT / 'gpu-tasks' / task['id']).resolve()
        if not directory.is_relative_to(ROOT / 'gpu-tasks'):
            raise ValueError('Invalid task ID')
        directory.mkdir(parents=True, exist_ok=True)
        try:
            inputs = self.inputs(task, directory)
            if task['kind'] == 'tokenize':
                response = self.local.post('/tokenize', json={'texts': task['payload']['texts']})
                response.raise_for_status()
                self.request('POST', '/result', json={'task_id': task['id'],
                    'lease_token': task['lease_token'], 'counts': response.json()['counts']})
            elif task['kind'] == 'synthesize':
                output = directory / 'result.wav'
                response = self.local.post('/synthesize', json={'text': task['payload']['text'],
                    'language': task['payload'].get('language', 'auto'),
                    'speaker_audio': inputs['speaker'], 'emotion_audio': inputs['emotion'],
                    'output': output.relative_to(ROOT).as_posix()})
                response.raise_for_status()
                self.upload(task, 'audio', output)
            elif task['kind'] == 'separate':
                response = self.local.post('/separations', json={'audio': inputs['audio'],
                    'prefix': (directory / 'demucs').relative_to(ROOT).as_posix(),
                    'model': task['payload']['model'], 'segment': task['payload']['segment']})
                response.raise_for_status()
                state = response.json()
                while state.get('status') in ('queued', 'running'):
                    if self.cancelled.wait(1):
                        self.local.delete('/separations/' + state['id'])
                        raise RuntimeError('GPU task lease was cancelled')
                    response = self.local.get('/separations/' + state['id'])
                    response.raise_for_status()
                    state = response.json()
                if state.get('status') != 'completed':
                    raise RuntimeError('Local Demucs failed')
                for name in task['outputs']:
                    self.upload(task, name, ROOT / state['stems'][name])
            else:
                raise ValueError('Unknown GPU task kind')
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def run(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        write_status(self.provider_id, ready=False, registered=False)
        self.request('POST', '/register', json={'provider_id': self.provider_id,
                                              'provider_type': self.provider_type})
        threading.Thread(target=self.heartbeat_loop, name='gpu-heartbeat', daemon=True).start()
        while not self.stopping.is_set():
            try:
                if not self.ready():
                    self.stopping.wait(5)
                    continue
                task = self.request('POST', '/claim').json().get('task')
                if not task:
                    self.stopping.wait(2)
                    continue
                self.cancelled.clear()
                with self.lock:
                    self.active = task
                try:
                    self.run_one(task)
                    print('GPU task completed:', task['kind'], task['id'], flush=True)
                except Exception as exc:
                    print('GPU task failed:', task['kind'], task['id'], type(exc).__name__, flush=True)
                    if not self.cancelled.is_set():
                        try:
                            self.request('POST', '/fail', json={'task_id': task['id'],
                                'lease_token': task['lease_token'], 'code': 'gpu_failed'})
                        except httpx.HTTPError:
                            pass
                finally:
                    with self.lock:
                        self.active = None
            except (httpx.HTTPError, ValueError) as exc:
                print('GPU agent connection unavailable:', type(exc).__name__, flush=True)
                self.stopping.wait(5)


if __name__ == '__main__':
    Agent().run()
