from dataclasses import replace

import httpx
from fastapi.testclient import TestClient

from videotranslator.api.app import create_app


def test_worker_gateway_keeps_worker_auth_and_streams_only_protocol_routes(container, monkeypatch):
    monkeypatch.setenv('GPU_BROKER_URL', 'http://gpu-broker:8082')
    requests = []
    def respond(request):
        requests.append((request.url.path, request.headers.get('authorization'), request.read()))
        if request.headers.get('authorization') != 'Bearer gpu-secret':
            return httpx.Response(401, json={'detail': 'Unauthorized GPU agent'})
        return httpx.Response(200, json={'id': 'gpu_one', 'type': 'local'})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(respond)))
    client = TestClient(create_app(container))
    assert client.post('/api/v1/gpu-workers/register', json={}).status_code == 401
    response = client.post('/api/v1/gpu-workers/register',
        headers={'Authorization': 'Bearer gpu-secret'}, json={'provider_id': 'gpu_one', 'provider_type': 'local'})
    assert response.status_code == 200 and response.json()['id'] == 'gpu_one'
    assert requests[-1][0] == '/api/v1/gpu-workers/register'
    assert b'gpu_one' in requests[-1][2]
    assert client.post('/api/v1/gpu-workers/synthesize', json={}).status_code == 404
    assert client.get('/api/v1/gpu-workers/health').status_code == 404


def test_provider_listing_admin_auth_and_internal_credential(container, monkeypatch):
    container.settings = replace(container.settings, admin_user_ids=('owner',))
    monkeypatch.setenv('ENGINE_CONTROL_URL', 'http://engine-control:8080')
    monkeypatch.setenv('ENGINE_CONTROL_TOKEN', 'internal-secret')
    def listing(url, headers, timeout):
        assert url == 'http://engine-control:8080/v1/gpu-providers'
        assert headers == {'Authorization': 'Bearer internal-secret'}
        return httpx.Response(200, json={'providers': [{'id': 'gpu_one', 'type': 'local'}]})
    monkeypatch.setattr(httpx, 'get', listing)
    client = TestClient(create_app(container))
    assert client.get('/api/v1/gpu-providers').status_code == 401
    assert client.get('/api/v1/gpu-providers', headers={'Authorization': 'Bearer fake:other'}).status_code == 403
    response = client.get('/api/v1/gpu-providers', headers={'Authorization': 'Bearer fake:owner'})
    assert response.status_code == 200 and response.json()['providers'][0]['id'] == 'gpu_one'
