"""Preview jobs must never charge for, or deliver, simulated media."""
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.domain.enums import DomainError, ErrorCode, JobStatus, LedgerEntryType
from videotranslator.domain.models import Job, JobOutbox, LedgerEntry, User
from .conftest import NOW, give_balance, make_job_via_inspection


@pytest.mark.parametrize('profile', ['local-ui', 'local-full'])
def test_preview_cannot_charge_or_enqueue(container, tmp_path, profile):
    settings = replace(container.settings, profile=profile,
        local_queue_db=str(tmp_path/'queue.db'))
    preview = build_container(settings)
    with preview.store.transaction() as tx:
        tx.insert(User(user_id='u1', point_balance_units=1000), 'u1')
    with pytest.raises(DomainError) as exc:
        make_job_via_inspection(preview)
    assert exc.value.code == ErrorCode.PROCESSING_UNAVAILABLE
    with preview.store.transaction() as tx:
        assert tx.get(User, 'u1').point_balance_units == 1000
        assert tx.query(Job) == tx.query(JobOutbox) == tx.query(LedgerEntry) == []
    assert preview.job_backend.name == 'unavailable'
    client = TestClient(create_app(preview))
    assert client.get('/api/v1/chat/config').json()['processing_available'] is False


def test_local_full_private_engine_is_available(container, tmp_path, monkeypatch):
    from unittest.mock import Mock
    from videotranslator.adapters.private_engine import PrivateEngineBackend

    backend = Mock(name='private-engine')
    monkeypatch.setattr(PrivateEngineBackend, 'from_env', lambda storage: backend)
    settings = replace(container.settings, profile='local-full', engine_backend='private',
                       local_queue_db=str(tmp_path / 'queue.db'))
    local = build_container(settings)
    backend.check_ready.assert_called_once_with()
    assert local.job_backend is backend
    assert local.settings.processing_available is True


def test_placeholder_recovery_returns_debit_once_and_blocks_download(container, user):
    give_balance(container, user.user_id, 1000)
    make_job_via_inspection(container)
    container.dispatcher.dispatch_due_jobs(now=NOW)
    for step in range(3):
        container.reconciler.reconcile_once(now=NOW + step + 1)
    with container.store.transaction() as tx:
        job = tx.get(Job, 'job_t1')
        assert job.status == JobStatus.SUCCEEDED
        assert container.storage.local_path(job.output_object_key).read_bytes() == b'fake-result'
    container.jobs._settings = replace(container.settings, profile='local-ui')
    assert container.jobs.recover_simulated_results() == 1
    assert container.jobs.recover_simulated_results() == 0
    with container.store.transaction() as tx:
        job = tx.get(Job, 'job_t1')
        assert job.status == JobStatus.FAILED and job.output_object_key is None
        assert job.error_code == 'processing_unavailable'
        assert tx.get(User, 'u1').point_balance_units == 1000
        assert len([e for e in tx.query(LedgerEntry) if e.entry_type == LedgerEntryType.JOB_REFUND]) == 1
    with pytest.raises(DomainError):
        container.jobs.download_url('job_t1', 'u1')


@pytest.mark.parametrize('backend,content', [('local-real', b'fake-result'), ('fake-0', b'real-media-bytes')])
def test_recovery_requires_fake_provenance_and_exact_placeholder(container, user, backend, content):
    give_balance(container, 'u1', 1000)
    make_job_via_inspection(container)
    with container.store.transaction() as tx:
        job = tx.get(Job, 'job_t1')
        job.status, job.backend_job_id, job.output_object_key = JobStatus.SUCCEEDED, backend, 'result.mp4'
        tx.put(job, job.job_id)
    container.storage.put_bytes('result.mp4', content)
    container.jobs._settings = replace(container.settings, profile='local-ui')
    assert container.jobs.recover_simulated_results() == 0
    with container.store.transaction() as tx:
        assert tx.get(Job, 'job_t1').status == JobStatus.SUCCEEDED
