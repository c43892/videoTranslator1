"""JobBackend adapter to the containerized Demucs/Whisper/IndexTTS2 engine."""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from ..domain.enums import BackendError, FailureClass
from ..domain.models import BackendJobRef, BackendStatus, JobSpec
from .ffmpeg import ffprobe_json, _run


class DockerEngineBackend:
    name = 'docker-engine'

    def __init__(self, storage, *, docker='docker', container='videotranslator-api-1'):
        self.storage, self.docker, self.container = storage, docker, container
        self._script = Path(__file__).with_name('engine_bridge.py').read_text(encoding='utf-8')
        self._lock = threading.RLock()

    @staticmethod
    def engine_id(key):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, 'videotranslator-studio:' + key))

    def _command(self, args, *, body=None, timeout=120):
        try:
            result = subprocess.run([self.docker, *args], input=body, text=True,
                encoding='utf-8', capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError('Container engine is unavailable', failure_class=FailureClass.RETRYABLE) from exc
        if result.returncode:
            # Do not persist container logs or provider secrets in user-facing errors.
            raise BackendError('Container engine command failed', failure_class=FailureClass.RETRYABLE)
        return result.stdout

    def _call(self, action, engine_id=None, **values):
        body = json.dumps(dict(action=action, id=engine_id, **values))
        return json.loads(self._command(['exec', '-i', self.container, 'python', '-c', self._script], body=body))

    def check_ready(self):
        if not self._call('health').get('ready'):
            raise BackendError('GPU engine configuration or models are not ready')

    def submit(self, spec: JobSpec, idempotency_key: str) -> BackendJobRef:
        engine_id = self.engine_id(idempotency_key)
        with self._lock:
            existing = self._call('status', engine_id)
            if existing['status'] != 'not_found':
                if existing['spec'] != spec.to_json_dict():
                    raise BackendError('Engine spec mismatch', failure_class=FailureClass.PERMANENT)
                return BackendJobRef(engine_id, engine_id)
            self.check_ready()
            source = self.storage.local_path(spec.input_uri.removeprefix('obj://'))
            if source is None or not source.is_file():
                raise BackendError('Input media is missing', failure_class=FailureClass.PERMANENT)
            with tempfile.TemporaryDirectory(prefix='vt-engine-input-') as directory:
                # The existing engine processes a video timeline. Audio-only inputs
                # get a private black video track; the delivered result stays audio.
                if spec.output_uri.endswith('.mp3'):
                    video = Path(directory) / 'input.mp4'
                    _run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                          'color=c=black:s=320x180:r=25', '-i', str(source), '-map', '0:v',
                          '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'ultrafast',
                          '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-t', str(spec.duration_ms / 1000),
                          '-shortest', str(video)], what='audio input preparation')
                    source = video
                self._command(['cp', str(source), f'{self.container}:/tmp/studio-{engine_id}.mp4'], timeout=600)
                self._call('submit', engine_id, spec=spec.to_json_dict())
        return BackendJobRef(engine_id, engine_id)

    def get_status(self, backend_job_id):
        try:
            engine_id = str(uuid.UUID(backend_job_id))
        except ValueError:
            engine_id = self.engine_id(backend_job_id)
        data = self._call('status', engine_id)
        state = data['status']
        if state in ('completed', 'completed_with_warnings'):
            try:
                with self._lock:
                    key = self._collect(data)
            except Exception as exc:
                if isinstance(exc, BackendError):
                    raise
                return BackendStatus(state='failed', error_message='Generated media failed playback validation')
            return BackendStatus(state='succeeded', progress_percent=100, output_object_key=key,
                                 warnings=data.get('warnings', []))
        if state in ('failed', 'needs_review'):
            return BackendStatus(state='failed', error_message=data.get('error') or 'Translation failed')
        if state in ('queued', 'waiting_configuration'):
            return BackendStatus(state='queued', stage='queued')
        if state in ('running', 'cancel_requested'):
            return BackendStatus(state='running', stage=data.get('stage', ''), progress_percent=data.get('progress', 0))
        if state in ('not_found', 'cancelled'):
            return BackendStatus(state=state)
        raise BackendError('Unexpected engine task state')

    def _collect(self, data):
        spec = data['spec']
        key = spec['output_uri'].removeprefix('obj://')
        if self.storage.exists(key):
            return key  # only atomically published, validated artifacts are cached
        audio = key.endswith('.mp3')
        with tempfile.TemporaryDirectory(prefix='vt-engine-result-') as directory:
            raw = Path(directory) / ('source.wav' if audio else 'source.mp4')
            source_path = data['outputs']['audio' if audio else 'video']
            if not source_path.startswith('/data/jobs/' + data['id'] + '/'):
                raise ValueError('Unexpected engine output location')
            self._command(['cp', f'{self.container}:{source_path}', str(raw)], timeout=600)
            output = Path(directory) / ('result.mp3' if audio else 'result.mp4')
            args = ['ffmpeg', '-v', 'error', '-y', '-i', str(raw)]
            if audio:
                args += ['-vn', '-c:a', 'libmp3lame', '-b:a', '192k']
            else:
                args += ['-map', '0:v:0', '-map', '0:a:0', '-map', '0:s?', '-c:s', 'mov_text', '-c:v', 'libx264',
                         '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
                         '-c:a', 'aac', '-movflags', '+faststart']
            _run(args + [str(output)], what='browser-compatible export')
            validate_output(output, spec['duration_ms'], audio=audio)
            # Native browser players use WebVTT; the downloadable MP4 retains its
            # independently switchable mov_text subtitle stream above.
            subtitle_path = data['outputs'].get('translated_vtt')
            if not audio and subtitle_path:
                if not subtitle_path.startswith('/data/jobs/' + data['id'] + '/'):
                    raise ValueError('Unexpected engine subtitle location')
                subtitles = Path(directory) / 'translated.vtt'
                self._command(['cp', f'{self.container}:{subtitle_path}', str(subtitles)], timeout=120)
                self.storage.upload(subtitles, key + '.vtt')
            final = self.storage.local_path(key)
            final.parent.mkdir(parents=True, exist_ok=True)
            # Atomic visibility: a crash during a copy never yields a successful job.
            import shutil
            staging = final.with_name(final.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                shutil.copyfile(output, staging)
                staging.replace(final)
            finally:
                staging.unlink(missing_ok=True)
        return key

    def cancel(self, backend_job_id):
        data = self._call('cancel', str(uuid.UUID(backend_job_id)))
        return data['status'] in ('cancelled', 'cancel_requested')


def validate_output(path, duration_ms, *, audio=False):
    info = ffprobe_json(path)
    streams = info.get('streams', [])
    types = {s.get('codec_type') for s in streams}
    if 'audio' not in types or (not audio and 'video' not in types):
        raise ValueError('Output is missing media streams')
    duration = float(info['format']['duration']) * 1000
    if duration <= 0 or abs(duration-duration_ms) > max(1500, duration_ms * .01):
        raise ValueError('Output duration does not match input')
    _run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], what='full output decode')
