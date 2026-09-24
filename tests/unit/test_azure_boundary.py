"""Cloud boundary guarantees: atomic media reads, SAS scope, and retry identity."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError

from videotranslator.adapters.azure_blob import AzureBlobStorage
from videotranslator.adapters.private_engine import PrivateEngineBackend
from videotranslator.domain.enums import BackendError
from .test_docker_engine import spec_and_outbox


def blob_storage():
    client = Mock()
    blob = client.get_blob_client.return_value
    blob.account_name = 'account'
    blob.url = 'https://account.blob.core.windows.net/uploads/user/input.mp4'
    blob.get_blob_properties.return_value = SimpleNamespace(etag='version-one', size=6)
    return AzureBlobStorage('https://account.blob.core.windows.net', client=client), client, blob


def test_blob_stream_is_atomic_and_version_bound(tmp_path):
    storage, client, blob = blob_storage()
    destination = tmp_path / 'input.mp4'
    destination.write_bytes(b'previous')
    blob.download_blob.return_value.chunks.return_value = iter([b'abc'])
    with pytest.raises(BackendError, match='size mismatch'):
        storage.download('inputs/user/input.mp4', destination)
    assert destination.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [destination]
    blob.download_blob.return_value.chunks.return_value = iter([b'abc', b'def'])
    storage.download('inputs/user/input.mp4', destination)
    assert destination.read_bytes() == b'abcdef'
    assert blob.download_blob.call_args.kwargs['etag'] == 'version-one'
    client.get_blob_client.assert_called_with('uploads', 'user/input.mp4')


def test_result_storage_is_separate_and_permission_errors_are_not_missing():
    storage, client, blob = blob_storage()
    storage.exists('outputs/user/result.mp4')
    client.get_blob_client.assert_called_with('results', 'user/result.mp4')
    with pytest.raises(ValueError):
        storage.create_upload_url('outputs/user/result.mp4', 300)
    for bad in ('../secret', 'user/../secret', '/secret', 'user\\secret', 'user//secret', 'user/file?sig=x'):
        with pytest.raises(ValueError):
            storage.exists(bad)
    blob.get_blob_properties.side_effect = ClientAuthenticationError('denied')
    with pytest.raises(ClientAuthenticationError):
        storage.exists('inputs/user/input.mp4')
    blob.get_blob_properties.side_effect = ResourceNotFoundError('missing')
    assert not storage.exists('inputs/user/input.mp4')


def test_sas_uses_delegation_key_and_only_requested_permission(monkeypatch):
    storage, client, blob = blob_storage()
    signer = Mock(return_value='signed')
    monkeypatch.setattr('azure.storage.blob.generate_blob_sas', signer)
    storage.create_upload_url('inputs/user/input.mp4', 300)
    permission = str(signer.call_args.kwargs['permission'])
    assert 'w' in permission and 'r' not in permission and 'd' not in permission
    assert signer.call_args.kwargs['protocol'] == 'https'
    storage.create_download_url('inputs/user/input.mp4', 300)
    assert str(signer.call_args.kwargs['permission']) == 'r'
    assert client.get_user_delegation_key.call_count == 1
    with pytest.raises(ValueError):
        storage.create_download_url('inputs/user/input.mp4', 3601)


def test_private_engine_replay_checks_spec_without_upload(container, user):
    spec, key = spec_and_outbox(container, user)
    calls = []
    def respond(request):
        calls.append(request)
        assert request.headers['authorization'] == 'Bearer ' + 'x' * 32
        return httpx.Response(200, json={'status': 'queued', 'spec': spec.to_json_dict()})
    client = httpx.Client(base_url='http://engine', transport=httpx.MockTransport(respond))
    backend = PrivateEngineBackend(container.storage, 'http://engine', 'x' * 32, client=client)
    assert backend.submit(spec, key).backend_job_id == backend.engine_id(key)
    with pytest.raises(BackendError, match='spec mismatch'):
        backend.submit(replace(spec, duration_ms=spec.duration_ms + 1), key)
    assert all(request.method == 'GET' for request in calls)


def test_control_health_does_not_request_gpu():
    paths = []
    def respond(request):
        paths.append(request.url.path)
        return httpx.Response(200, json={'ready': True})
    client = httpx.Client(base_url='http://engine', transport=httpx.MockTransport(respond))
    PrivateEngineBackend(None, 'http://engine', 'x' * 32, client=client).check_ready()
    assert paths == ['/health']


def test_cpu_profile_runs_processors_without_gpu_readiness(container, monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    from videotranslator.api.app import create_app
    container.settings = replace(container.settings, profile='azure-jp-t4')
    processed = threading.Event()
    monkeypatch.setattr(container.reconciler, 'reconcile_once', lambda: processed.set())
    with TestClient(create_app(container)) as client:
        assert client.get('/api/v1/health/ready').status_code == 200
        assert processed.wait(3), 'CPU scheduler must keep running while GPU is zero'
