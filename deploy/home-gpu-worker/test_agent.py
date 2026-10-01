"""Protocol test: the real agent transfers media through the broker contract."""
import importlib.util
import io
import json
import os
import tempfile
import threading
from pathlib import Path

_test_root = Path(tempfile.mkdtemp(prefix='vt-gpu-agent-test-'))
os.environ['DATABASE_URL'] = 'sqlite:///' + (_test_root / 'broker.db').as_posix()
os.environ['STORAGE_ROOT'] = str(_test_root / 'storage')

import httpx
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from videotranslator.db import Base, engine
from videotranslator.gpu_broker import app, storage

AGENT_FILE = Path(__file__).with_name('agent.py')
spec = importlib.util.spec_from_file_location('home_gpu_agent', AGENT_FILE)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
TOKEN = 'gpu-host-test-token-' + 'x' * 32
INTERNAL = 'internal-test-token-' + 'y' * 32


def wav():
    out = io.BytesIO()
    sf.write(out, np.zeros(1600, dtype='float32'), 16000, format='WAV')
    return out.getvalue()


def test_agent_completes_speech_and_demucs_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv('GPU_WORKER_TOKENS', TOKEN)
    monkeypatch.setenv('GPU_BROKER_INTERNAL_TOKEN', INTERNAL)
    monkeypatch.setattr(agent_module, 'ROOT', tmp_path)
    Base.metadata.drop_all(engine)
    broker_storage = storage()
    for name in ('speaker', 'emotion', 'source'):
        key = f'jobs/agent-test/{name}.wav' if name != 'source' else 'jobs/agent-test/source.flac'
        path = broker_storage.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav())

    def local_tts(request):
        body = json.loads(request.content) if request.content else {}
        if request.url.path == '/health':
            result = {'status': 'ready'}
        elif request.url.path == '/synthesize':
            assert body['language'] == 'es'
            output = tmp_path / body['output']
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(wav())
            result = {'output': body['output']}
        elif request.url.path == '/separations':
            stems = {}
            for name in ('dialogue', 'music', 'effects'):
                key = body['prefix'] + f'/{name}.wav'
                output = tmp_path / key
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(wav())
                stems[name] = key
            result = {'id': 'local-separation', 'status': 'completed', 'stems': stems}
        elif request.url.path == '/tokenize':
            result = {'counts': [1 for _ in body['texts']]}
        else:
            return httpx.Response(404)
        return httpx.Response(200, json=result)

    with TestClient(app) as broker:
        def remote(request):
            response = broker.request(request.method, request.url.path,
                                      headers=dict(request.headers), content=request.read())
            return httpx.Response(response.status_code, headers=dict(response.headers),
                                  content=response.content)

        agent = agent_module.Agent.__new__(agent_module.Agent)
        agent.remote = httpx.Client(base_url='https://test.example',
            headers={'Authorization': 'Bearer ' + TOKEN}, transport=httpx.MockTransport(remote))
        agent.local = httpx.Client(base_url='http://tts', transport=httpx.MockTransport(local_tts))
        agent.cancelled = threading.Event()
        agent.lock = threading.Lock()
        agent.active = None
        heartbeat = agent.request('POST', '/heartbeat',
            json={'ready': True, 'gpu_active': False, 'active': []})
        assert heartbeat.status_code == 200

        # The two internal requests create durable tasks without awaiting a GPU.
        from videotranslator.gpu_broker import _create
        _create('synthesize', {'text': 'hello', 'language': 'es',
            'inputs': {'speaker': 'jobs/agent-test/speaker.wav',
                       'emotion': 'jobs/agent-test/emotion.wav'},
            'outputs': {'audio': 'jobs/agent-test/result.wav'}}, 60)
        speech = agent.request('POST', '/claim').json()['task']
        assert speech['kind'] == 'synthesize'
        agent.run_one(speech)
        assert sf.info(str(broker_storage.path('jobs/agent-test/result.wav'))).frames > 0

        _create('separate', {'model': 'htdemucs', 'segment': 5,
            'inputs': {'audio': 'jobs/agent-test/source.flac'},
            'outputs': {name: f'jobs/agent-test/demucs/{name}.wav'
                        for name in ('dialogue', 'music', 'effects')}}, 60)
        separation = agent.request('POST', '/claim').json()['task']
        assert separation['kind'] == 'separate'
        agent.run_one(separation)
        for name in ('dialogue', 'music', 'effects'):
            assert sf.info(str(broker_storage.path(f'jobs/agent-test/demucs/{name}.wav'))).frames > 0
