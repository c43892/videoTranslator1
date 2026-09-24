"""History uses durable job records, with the same ownership as job operations."""
from dataclasses import replace

from fastapi.testclient import TestClient
import pytest

from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.domain.enums import JobStatus
from videotranslator.domain.models import Job
from videotranslator.domain.models import User, LedgerEntry, JobOutbox
from .conftest import make_job_via_inspection


def test_history_is_owned_and_reads_current_status(container):
    with container.store.transaction() as tx:
        tx.insert(Job(job_id="mine",owner_user_id="alice",original_filename="speech.wav",status=JobStatus.QUEUED,created_at=123),"mine")
        tx.insert(Job(job_id="private",owner_user_id="bob",original_filename="private.mp4"),"private")
    client=TestClient(create_app(container),headers={"Authorization":"Bearer fake:alice"})
    jobs=client.get("/api/v1/jobs").json()["jobs"]
    assert [job["job_id"] for job in jobs] == ["mine"]
    assert jobs[0]["status"] == "queued"
    with container.store.transaction() as tx:
        job=tx.get(Job,"mine")
        job.status,job.progress_percent=JobStatus.RUNNING,42
        tx.put(job,"mine")
    jobs=client.get("/api/v1/jobs").json()["jobs"]
    assert jobs[0]["status"] == "running" and jobs[0]["progress_percent"] == 42
    assert client.get("/api/v1/jobs/private").status_code == 403
    assert TestClient(client.app).get("/api/v1/jobs").status_code == 401


def test_history_survives_restart_and_asset_deletion(container,tmp_path):
    settings=replace(container.settings,store_path=str(tmp_path / "history.db"))
    first=build_container(settings)
    with first.store.transaction() as tx:
        tx.insert(Job(job_id="done",owner_user_id="alice",status=JobStatus.SUCCEEDED,
            original_filename="recording.mp3",media_type="audio",created_at=123,completed_at=456,
            input_object_key="input.mp3",output_object_key="result.mp3",quoted_point_units=15),"done")
        tx.insert(Job(job_id="error",owner_user_id="alice",status=JobStatus.FAILED,
            error_code="backend_failed",created_at=234,completed_at=567),"error")
    first.jobs.delete_assets("done","alice")
    restored=build_container(settings)
    client=TestClient(create_app(restored),headers={"Authorization":"Bearer fake:alice"})
    jobs={row["job_id"]:row for row in client.get("/api/v1/jobs").json()["jobs"]}
    assert set(jobs) == {"done","error"}
    assert jobs["done"]["status"] == "succeeded" and jobs["done"]["assets_deleted_at"]
    assert jobs["done"]["quoted_point_units"] == 15 and jobs["done"]["completed_at"] == 456
    assert jobs["error"]["status"] == "failed" and jobs["error"]["error_code"] == "backend_failed"


@pytest.mark.parametrize('state', [JobStatus.QUEUED, JobStatus.PROVISIONING, JobStatus.SUCCEEDED])
def test_persisted_start_and_inspection_replay_do_not_charge_twice(container, tmp_path, state):
    settings = replace(container.settings, store_path=str(tmp_path / 'replay.db'))
    first = build_container(settings)
    first.seed()
    with first.store.transaction() as tx:
        tx.insert(User(user_id='u1', point_balance_units=1000), 'u1')
    original = make_job_via_inspection(first).job
    with first.store.transaction() as tx:
        job = tx.get(Job, original.job_id)
        job.status = state
        tx.put(job, job.job_id)
        balance = tx.get(User, 'u1').point_balance_units
    restored = build_container(settings)
    client = TestClient(create_app(restored), headers={'Authorization':'Bearer fake:u1'})
    response = client.post(f'/api/v1/jobs/{original.job_id}/start')
    assert response.status_code == 200 and response.json()['outcome'] == str(state)
    replay = make_job_via_inspection(restored, duration_ms=120000)
    assert replay.outcome == str(state) and replay.job.duration_ms == 60000
    with restored.store.transaction() as tx:
        assert tx.get(User, 'u1').point_balance_units == balance
        assert len(tx.query(LedgerEntry)) == 1
        assert len(tx.query(JobOutbox)) == 1
