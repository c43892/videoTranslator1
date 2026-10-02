#!/usr/bin/env python3
"""Deploy one local GPU mode using Docker Compose; no third-party Python packages."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {'demo': 'videotranslator-local-gpu', 'provider': 'videotranslator-home-gpu'}
GPU_HEALTH = '''import json,urllib.request,urllib.error
try:
 r=urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=5)
except urllib.error.HTTPError as e:
 r=e
print(r.read().decode())
'''


@contextmanager
def mode_lock():
    # Shared across repository clones; OS locks are released even after a crash.
    path = Path.home() / '.cache' / 'videotranslator' / 'local-mode.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0)
        stream.write(b'0')
        stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another deployment switch is already running.') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def initialize():
    for mode, name in [('demo', '.env.local-gpu'), ('provider', '.env.home-gpu')]:
        path = ROOT / name
        if path.exists():
            print(f'Keeping existing {path.name}')
            continue
        contents = (ROOT / (name + '.example')).read_text(encoding='utf-8')
        if mode == 'demo':
            for key in ['POSTGRES_PASSWORD', 'ENGINE_CONTROL_TOKEN',
                        'LOCAL_STORAGE_SIGNING_SECRET', 'GPU_BROKER_INTERNAL_TOKEN']:
                contents = contents.replace(key + '=\n', key + '=' + secrets.token_urlsafe(48) + '\n')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(contents)
        print(f'Created private {path.name}')
    print('Demo: set OPENAI_API_KEY (and DEEPSEEK_API_KEY for chat) in .env.local-gpu.')
    print('Provider: set its cloud URL and enrolled GPU_WORKER_TOKEN in .env.home-gpu.')


class Deployment:
    def __init__(self, docker='docker', demo_env=None, provider_env=None, timeout=1800):
        self.docker = docker
        self.envs = {'demo': Path(demo_env or ROOT / '.env.local-gpu').resolve(),
                     'provider': Path(provider_env or ROOT / '.env.home-gpu').resolve()}
        self.timeout = timeout

    def run(self, args, *, capture=False, check=True, timeout=None):
        result = subprocess.run([self.docker, *args], cwd=ROOT, text=True,
                                capture_output=capture, timeout=timeout)
        if check and result.returncode:
            # Compose config contains credentials: never print its captured stdout.
            raise RuntimeError(result.stderr.strip() if capture else
                               f'Docker command failed (exit {result.returncode}).')
        return result

    def compose(self, mode, *args, **kwargs):
        files = ['compose.local-gpu.yml', 'compose.demo.yml'] if mode == 'demo' else ['compose.home-gpu.yml']
        command = ['compose', '--env-file', str(self.envs[mode])]
        for name in files:
            command.extend(['-f', str(ROOT / name)])
        return self.run([*command, *args], **kwargs)

    def configuration(self, mode):
        if not self.envs[mode].is_file():
            raise RuntimeError(f'Missing {self.envs[mode].name}; run init and configure it first.')
        cfg = json.loads(self.compose(mode, 'config', '--format', 'json', capture=True).stdout)
        services = cfg['services']
        if mode == 'demo':
            web = services['studio']['environment']
            control = services['engine-control']['environment']
            if (web['AUTH_MODE'] != 'demo' or web['PAYMENT_MODE'] != 'disabled'
                    or control['GPU_PROVIDER_MODE'] != 'local_only'
                    or str(control['GPU_AZURE_T4_ENABLED']).lower() != 'false'):
                raise RuntimeError('Demo must be free, unauthenticated and local GPU only.')
            if not control.get('OPENAI_API_KEY'):
                raise RuntimeError('Demo needs OPENAI_API_KEY for Whisper and translation.')
        else:
            env = services['agent']['environment']
            url = urlsplit(env.get('GPU_AGENT_SERVER_URL', ''))
            token = env.get('GPU_WORKER_TOKEN', '')
            if url.scheme != 'https' or not url.hostname or url.username or url.password:
                raise RuntimeError('GPU_AGENT_SERVER_URL must be an HTTPS URL without credentials.')
            if len(token) < 32 or token.startswith('replace-'):
                raise RuntimeError('Set the cloud-enrolled GPU_WORKER_TOKEN, at least 32 characters.')
        return cfg

    def container_ids(self, mode):
        result = self.run(['ps', '-aq', '--filter', 'label=com.docker.compose.project=' + PROJECTS[mode]], capture=True)
        return result.stdout.split()

    def stop_mode(self, mode):
        ids = self.container_ids(mode)
        if ids:
            print(f'Stopping {mode}; Docker volumes and results will be retained.', flush=True)
            self.run(['stop', '--time', '30', *ids])
            self.run(['rm', *ids])  # No -v: never delete task/model/database volumes.
        if self.container_ids(mode):
            raise RuntimeError(f'{mode} containers still exist; refusing to start another GPU backend.')

    def wait_json(self, container, code, predicate, label):
        deadline = time.monotonic() + self.timeout
        last = None
        while time.monotonic() < deadline:
            result = self.run(['exec', container, 'python', '-c', code], capture=True, check=False, timeout=20)
            try:
                state = json.loads(result.stdout) if result.returncode == 0 else {}
            except ValueError:
                state = {}
            status = state.get('status', 'waiting')
            if status != last:
                print(f'{label}: {status}', flush=True)
                last = status
            if state.get('status') == 'error':
                raise RuntimeError(f'{label} failed; inspect its Docker logs.')
            if predicate(state):
                return state
            time.sleep(5)
        raise RuntimeError(f'{label} timed out. The opposite mode remains stopped; inspect Docker logs.')

    def start(self, mode, build=False):
        cfg = self.configuration(mode)  # Validate before stopping the currently selected mode.
        self.run(['info', '--format', '{{.ServerVersion}}'], capture=True)
        if not build:
            for image in {s['image'] for s in cfg['services'].values() if s.get('build')}:
                result = self.run(['image', 'inspect', image], capture=True, check=False)
                if result.returncode:
                    raise RuntimeError(f'Missing image {image}; rerun {mode} with --build.')
        self.stop_mode('provider' if mode == 'demo' else 'demo')
        if build:
            self.compose(mode, 'build')
        for volume in cfg.get('volumes', {}).values():
            if volume.get('external'):
                self.run(['volume', 'create', volume['name']], capture=True)
        # Load the GPU before enabling web admission or cloud task claims.
        self.compose(mode, 'up', '-d', '--no-build', 'tts')
        gpu = PROJECTS[mode] + '-tts-1'
        state = self.wait_json(gpu, GPU_HEALTH, lambda s: s.get('status') == 'ready', 'GPU')
        print('GPU ready:', state.get('model', ''), flush=True)
        self.compose(mode, 'up', '-d', '--no-build')
        if mode == 'provider':
            code = 'import json,time; from pathlib import Path; p=Path("/tmp/gpu-agent-status.json"); print(p.read_text() if p.exists() else "{}")'
            state = self.wait_json(PROJECTS[mode] + '-agent-1', code,
                lambda s: bool(s.get('registered') and s.get('ready') and
                               0 <= time.time() - s.get('acknowledged_at', 0) < 30), 'Cloud heartbeat')
            print('Cloud GPU provider ready:', state['provider_id'])
        else:
            code = 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8000/api/v1/chat/config",timeout=5).read().decode())'
            self.wait_json(PROJECTS[mode] + '-studio-1', code,
                           lambda s: s.get('auth_mode') == 'demo' and
                           s.get('billing_enabled') is False and s.get('processing_available') is True,
                           'Demo App')
            print('Local Demo ready: http://127.0.0.1:8090/ (no cloud GPU registration)')

    def status(self):
        for mode in PROJECTS:
            print(f'[{mode}]', flush=True)
            self.run(['ps', '-a', '--filter', 'label=com.docker.compose.project=' + PROJECTS[mode],
                      '--format', 'table {{.Names}}\t{{.Status}}'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['init', 'demo', 'provider', 'stop', 'status'])
    parser.add_argument('--build', action='store_true', help='Build images on a new host or after pulling code updates.')
    parser.add_argument('--docker', default='docker', help='Docker executable path if it is not on PATH.')
    parser.add_argument('--demo-env', help='Private Demo env file (default .env.local-gpu).')
    parser.add_argument('--provider-env', help='Private GPU provider env file (default .env.home-gpu).')
    parser.add_argument('--timeout', type=int, default=1800, help='Readiness timeout in seconds (default 1800).')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    try:
        with mode_lock():
            if args.mode == 'init':
                initialize()
            else:
                deployment = Deployment(args.docker, args.demo_env, args.provider_env, args.timeout)
                if args.mode == 'status':
                    deployment.status()
                elif args.mode == 'stop':
                    for mode in PROJECTS:
                        deployment.stop_mode(mode)
                else:
                    deployment.start(args.mode, args.build)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
