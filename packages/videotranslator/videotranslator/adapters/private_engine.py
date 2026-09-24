"""Studio boundary for the private, durable cloud engine (no Docker socket)."""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path

import httpx

from .docker_engine import DockerEngineBackend, validate_output
from .ffmpeg import _run
from ..domain.enums import BackendError, FailureClass
from ..domain.models import BackendJobRef, BackendStatus


class PrivateEngineBackend:
    name = 'private-engine'
    engine_id = staticmethod(DockerEngineBackend.engine_id)

    def __init__(self, storage, url, token, *, client=None):
        if not url.startswith(('http://', 'https://')) or len(token) < 32:
            raise ValueError('Private engine URL and a token of at least 32 characters are required')
        self.storage = storage
        self.client = client or httpx.Client(base_url=url.rstrip('/'), timeout=httpx.Timeout(30, read=600, write=600))
        self.headers = {'Authorization': 'Bearer ' + token}
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, storage):
        return cls(storage, os.environ['ENGINE_CONTROL_URL'], os.environ['ENGINE_CONTROL_TOKEN'])

    @staticmethod
    def _check(response):
        if response.is_error:
            retryable = response.status_code >= 500 or response.status_code in (408, 429)
            raise BackendError('Private engine request failed', failure_class=(
                FailureClass.RETRYABLE if retryable else FailureClass.PERMANENT))

    def _call(self, action, identifier=None):
        path = '/health' if action == 'health' else '/v1/jobs/' + str(uuid.UUID(identifier))
        try:
            response = self.client.request('DELETE' if action == 'cancel' else 'GET', path, headers=self.headers)
            if response.status_code == 404 and action == 'status':
                return {'status': 'not_found'}
            self._check(response)
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError('Private engine is unavailable', failure_class=FailureClass.RETRYABLE) from exc

    def check_ready(self):
        # Control health never calls the GPU: zero replicas is a healthy idle state.
        if not self._call('health').get('ready'):
            raise BackendError('Private engine control is unavailable')

    def submit(self, spec, idempotency_key):
        identifier = self.engine_id(idempotency_key)
        existing = self._call('status', identifier)
        if existing['status'] != 'not_found':
            if existing['spec'] != spec.to_json_dict():
                raise BackendError('Engine spec mismatch', failure_class=FailureClass.PERMANENT)
            return BackendJobRef(identifier, identifier)
        with tempfile.TemporaryDirectory(prefix='vt-private-input-') as directory:
            key = spec.input_uri.removeprefix('obj://')
            source = self.storage.local_path(key) or self.storage.download(key, Path(directory) / 'input')
            if spec.output_uri.endswith('.mp3'):
                video = Path(directory) / 'input.mp4'
                _run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=320x180:r=25',
                      '-i', str(source), '-map', '0:v', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'ultrafast',
                      '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-t', str(spec.duration_ms / 1000), '-shortest', str(video)],
                     what='audio input preparation')
                source = video
            headers = {**self.headers, 'X-Job-Spec': base64.b64encode(json.dumps(spec.to_json_dict()).encode()).decode(),
                       'Content-Type': 'application/octet-stream', 'Content-Length': str(source.stat().st_size)}
            try:
                with source.open('rb') as stream:
                    response = self.client.put('/v1/jobs/' + identifier, headers=headers,
                                               content=iter(lambda: stream.read(1024 * 1024), b''))
                self._check(response)
            except httpx.HTTPError as exc:
                raise BackendError('Private engine submission interrupted', failure_class=FailureClass.RETRYABLE) from exc
        return BackendJobRef(identifier, identifier)

    def get_status(self, identifier):
        data = self._call('status', identifier)
        state = data['status']
        if state in ('completed', 'completed_with_warnings'):
            with self._lock:
                key = self._collect(data)
            return BackendStatus(state='succeeded', progress_percent=100, output_object_key=key,
                                 warnings=data.get('warnings', []), actual_gpu_seconds=data.get('runtime_seconds'))
        if state in ('failed', 'needs_review', 'waiting_configuration'):
            return BackendStatus(state='failed', error_message=data.get('error') or 'Engine configuration is unavailable')
        if state in ('queued', 'provisioning'):
            return BackendStatus(state='provisioning', stage='starting_gpu')
        if state in ('running', 'cancel_requested'):
            return BackendStatus(state='running', stage=data.get('stage', ''), progress_percent=data.get('progress', 0))
        if state in ('not_found', 'cancelled'):
            return BackendStatus(state=state)
        raise BackendError('Unexpected private engine state')

    def _download_artifact(self, identifier, kind, destination):
        try:
            with self.client.stream('GET', f'/v1/jobs/{identifier}/artifacts/{kind}', headers=self.headers) as response:
                self._check(response)
                with destination.open('wb') as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
        except httpx.HTTPError as exc:
            raise BackendError('Artifact transfer interrupted', failure_class=FailureClass.RETRYABLE) from exc

    def _collect(self, data):
        identifier = str(uuid.UUID(data['id']))
        generation = str(uuid.UUID(data['generation']))
        spec = data['spec']
        original = Path(spec['output_uri'].removeprefix('obj://'))
        key = f'outputs/{original.parent.as_posix()}/{identifier}/{generation}/{original.name}'
        manifest_key = key + '.manifest.json'
        with tempfile.TemporaryDirectory(prefix='vt-private-result-') as directory:
            root = Path(directory)
            manifest = root / 'published.json'
            if self.storage.exists(manifest_key):
                self.storage.download(manifest_key, manifest)
                saved = json.loads(manifest.read_text())
                if (saved.get('key') == key and saved.get('generation') == generation
                        and self.storage.exists(key, expected_size=saved['size'])
                        and (not saved.get('subtitles') or self.storage.exists(key + '.vtt', expected_size=saved['subtitles']))):
                    return key
            audio = original.suffix == '.mp3'
            raw = root / ('source.wav' if audio else 'source.mp4')
            self._download_artifact(identifier, 'audio' if audio else 'video', raw)
            output = root / original.name
            args = ['ffmpeg', '-v', 'error', '-y', '-i', str(raw)]
            if audio:
                args += ['-vn', '-c:a', 'libmp3lame', '-b:a', '192k']
            else:
                args += ['-map', '0:v:0', '-map', '0:a:0', '-map', '0:s?', '-c:s', 'mov_text',
                         '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                         '-c:a', 'aac', '-movflags', '+faststart']
            _run(args + [str(output)], what='browser-compatible export')
            validate_output(output, spec['duration_ms'], audio=audio)
            subtitle_size = 0
            if not audio and data.get('outputs', {}).get('translated_vtt'):
                subtitle = root / 'translated.vtt'
                self._download_artifact(identifier, 'translated_vtt', subtitle)
                subtitle_size = subtitle.stat().st_size
                self.storage.upload(subtitle, key + '.vtt')
            current = self._call('status', identifier)
            if current.get('generation') != generation or current['status'] not in ('completed', 'completed_with_warnings'):
                raise BackendError('Engine result changed before publication')
            self.storage.upload(output, key)
            manifest.write_text(json.dumps({'key': key, 'generation': generation,
                                           'size': output.stat().st_size, 'subtitles': subtitle_size}))
            self.storage.upload(manifest, manifest_key)
        return key

    def cancel(self, identifier):
        return self._call('cancel', identifier)['status'] in ('cancel_requested', 'cancelled')
