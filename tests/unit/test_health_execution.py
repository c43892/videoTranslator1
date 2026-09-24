from videotranslator.domain.models import Job, JobOutbox
from .conftest import give_balance, make_job_via_inspection


def test_health_policy_keeps_budget_reservation_without_total_deadline(container, user):
    container.funding.health_supervised = True
    give_balance(container, user.user_id, 1000)
    result = make_job_via_inspection(container)
    with container.store.transaction() as tx:
        job = tx.get(Job, 'job_t1')
        outbox = tx.query(JobOutbox)[0]
        assert job.execution_deadline_at is None
        assert job.max_runtime_seconds > 0 and job.cost_reservation_id
        assert outbox.jobspec['processing_profile'] == 'health-v1'


def test_web_does_not_cancel_supervised_backend_at_old_deadline(container, monkeypatch):
    monkeypatch.setattr(container.job_backend, 'supervises_execution', True, raising=False)
    def cancel(_): raise AssertionError('healthy engine must not be cancelled by a total deadline')
    monkeypatch.setattr(container.job_backend, 'cancel', cancel)
    job = Job(job_id='test', status='running', backend_job_id='engine', execution_deadline_at=1)
    assert container.reconciler._enforce_deadline(job, now=10_000_000) == 0
