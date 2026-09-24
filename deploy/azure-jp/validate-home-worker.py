"""Attended, GPU-free HTTPS acceptance test with a dedicated synthetic identity.

Uses private server/worker configuration already present in secrets/. It never
confirms translation, changes credits, or logs credentials.
"""
from pathlib import Path
import importlib.util
import json
import tempfile
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    identity = load("validation_identity", ROOT / 'deploy/azure-jp/validate-web.py')
    worker = load("home_worker", ROOT / 'deploy/home-download-worker/worker.py')
    config = worker.load_config(ROOT / 'secrets/home-download-worker/config.json')
    check = identity.check
    evidence = {"url": "https://www.youtube.com/watch?v=DlldBRDJXE4", "gpu_started": False}
    with identity.client() as api:
        assert httpx.post(config['server_url']+'/api/v1/download-workers/claim', timeout=15).status_code == 401
        before_balance = check(api.get('/api/v1/me'))['point_balance_units']

        def prepare_new():
            draft = check(api.post('/api/v1/conversations', json={'locale': 'en'}), 201)
            key = draft['conversation_id']
            draft = check(api.post(f'/api/v1/conversations/{key}/messages', json={
                'revision': draft['revision'], 'text': 'Translate to English '+evidence['url']}))
            assert draft['ready']
            return check(api.post(f'/api/v1/conversations/{key}/prepare', json={'revision':draft['revision']}))

        offline = prepare_new()
        key = offline['conversation_id']
        start = time.monotonic()
        while time.monotonic()-start < 155:
            offline = check(api.get(f'/api/v1/conversations/{key}'))
            if offline['status'] == 'import_failed':
                break
            time.sleep(2)
        assert offline['status']=='import_failed' and offline['error']=='youtube_proxy_unavailable', offline['status']
        assert offline['download']['unavailable_retries']==10
        evidence['offline_retry_seconds'] = round(time.monotonic()-start,1)
        evidence['offline_retry_count'] = 10
        print('Cloud offline test passed: ten spaced retries, then retryable failure.', flush=True)

        retry = check(api.post(f'/api/v1/conversations/{key}/prepare', json={'revision':offline['revision']}))
        assert retry['download']['unavailable_retries']==0
        drafts = [retry, prepare_new(), prepare_new()]
        keys = [d['conversation_id'] for d in drafts]
        evidence['conversation_ids'] = keys
        observed_two = observed_waiting = False
        with tempfile.TemporaryDirectory(prefix='vt-home-cloud-', dir=ROOT/'vt-data') as directory:
            agent = worker.Agent(config, root=Path(directory))
            thread = threading.Thread(target=agent.run, daemon=True)
            start = time.monotonic()
            thread.start()
            try:
                while time.monotonic()-start < 300:
                    drafts = [check(api.get(f'/api/v1/conversations/{key}')) for key in keys]
                    states = [d.get('download',{}).get('status') for d in drafts]
                    active = sum(state=='leased' for state in states)
                    assert active <= 2, states
                    observed_two |= active == 2
                    observed_waiting |= active == 2 and 'queued' in states
                    if all(d['status']=='draft' and d['duration_ms'] for d in drafts):
                        break
                    assert all(d['status']!='import_failed' for d in drafts), [d['error'] for d in drafts]
                    time.sleep(1)
                assert all(d['status']=='draft' and 71000<d['duration_ms']<73000 for d in drafts)
                assert observed_two and observed_waiting
                assert check(api.get('/api/v1/me'))['point_balance_units']==before_balance
                evidence.update(status='passed', observed_two_concurrent=True, observed_third_queued=True,
                    download_seconds=round(time.monotonic()-start,1),
                    durations_ms=[d['duration_ms'] for d in drafts], balance_unchanged=True)
                print(json.dumps(evidence), flush=True)
            finally:
                agent.stopping.set()
                thread.join(timeout=45)
                assert not thread.is_alive(), 'Worker did not stop'
        output = ROOT / 'vt-data/youtube-diagnostics/home-worker-cloud.json'
        output.write_text(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
