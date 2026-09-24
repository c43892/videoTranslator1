"""Failed tasks return spendable balance exactly once, never payment-provider cash."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from videotranslator.bootstrap import build_container
from videotranslator.config import Settings
from videotranslator.domain.enums import ErrorCode, JobStatus, LedgerEntryType, RefundStatus
from videotranslator.domain.models import Job, LedgerEntry, Payment, User

from .conftest import NOW, give_balance, make_job_via_inspection


def charged_job(container, user):
    container.funding._pricing = Settings().pricing
    give_balance(container,user.user_id,1000)
    make_job_via_inspection(container,user.user_id,duration_ms=90000)
    container.dispatcher.dispatch_due_jobs(now=NOW)


def test_concurrent_failure_returns_actual_debit_once_and_can_be_spent(container,user):
    charged_job(container,user)
    with container.store.transaction() as tx:
        job=tx.get(Job,'job_t1')
        job.quoted_point_units=999  # A quote edit cannot inflate the return.
        tx.put(job,job.job_id)
    def fail(_):
        return container.funding.fail_and_refund('job_t1',error_code=ErrorCode.BACKEND_FAILED,
            refund=False,now=NOW+1)  # Legacy callers cannot suppress the new rule.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fail,range(4)))
    assert all(job.status == JobStatus.FAILED and job.balance_returned_cents == 30 for job in results)
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 1000
        entries=tx.query(LedgerEntry)
        assert len([e for e in entries if e.entry_type == LedgerEntryType.JOB_REFUND]) == 1
        assert sum(e.delta_units for e in entries) == 0
    make_job_via_inspection(container,user.user_id,duration_ms=90000,job_id='next-task',now=NOW+2)
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 970
        assert tx.get(Job,'next-task').status == JobStatus.QUEUED


def test_no_cash_refund_or_payment_reversal(container,user):
    session=container.billing.create_session(user.user_id,package_id='points_10_v1',provider='stripe',now=NOW)
    headers,body=container.gateways['stripe'].make_webhook(session.payment_id)
    container.billing.record_webhook('stripe',headers,body,now=NOW)
    container.inbox.process_due(now=NOW)
    for gateway in container.gateways.values():
        def forbidden(*args,**kwargs):
            raise AssertionError('A failed task must never call a payment gateway refund')
        gateway.refund=forbidden
    container.funding._pricing = Settings().pricing
    make_job_via_inspection(container,user.user_id,duration_ms=90000)
    container.dispatcher.dispatch_due_jobs(now=NOW)
    container.funding.fail_and_refund('job_t1',error_code=ErrorCode.EXECUTION_TIMEOUT,now=NOW+1)
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 1000
        assert tx.get(Payment,session.payment_id).status == Payment.STATUS_SUCCEEDED
        assert not [e for e in tx.query(LedgerEntry) if e.entry_type == LedgerEntryType.PAYMENT_REVERSAL]


def test_uncharged_inspection_does_not_create_credit(container,user):
    with container.store.transaction() as tx:
        tx.insert(Job(job_id='inspection',owner_user_id=user.user_id,status=JobStatus.INSPECTING),'inspection')
    failed=container.funding.fail_and_refund('inspection',error_code=ErrorCode.MEDIA_INSPECTION_FAILED,now=NOW)
    assert failed.refund_status == RefundStatus.NOT_APPLICABLE and failed.balance_returned_cents == 0
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 0
        assert tx.query(LedgerEntry) == []


def test_legacy_failed_debit_recovered_after_restart(container,tmp_path):
    old_price=replace(Settings().pricing,pricing_version='usd-cent-v1',point_units_per_minute=10,
                      minimum_point_units=1,billing_increment_units=1,rounding='ceil_final_point_unit')
    settings=replace(container.settings,store_path=str(tmp_path/'returns.db'),pricing=old_price)
    original=build_container(settings)
    with original.store.transaction() as tx:
        tx.insert(User(user_id='u1',point_balance_units=1000),'u1')
    make_job_via_inspection(original,'u1',duration_ms=90000)
    with original.store.transaction() as tx:
        job=tx.get(Job,'job_t1')
        job.status,job.error_message,job.completed_at=JobStatus.FAILED,'Original failure',NOW
        tx.put(job,job.job_id)
    restarted=build_container(replace(settings,pricing=Settings().pricing))
    restarted.reconciler.reconcile_once(now=NOW+1)
    restarted.reconciler.reconcile_once(now=NOW+2)
    with restarted.store.transaction() as tx:
        job=tx.get(Job,'job_t1')
        assert job.error_message == 'Original failure' and job.completed_at == NOW
        assert job.balance_returned_cents == 15 and job.balance_returned_at == NOW+1
        assert tx.get(User,'u1').point_balance_units == 1000
        assert len([e for e in tx.query(LedgerEntry) if e.entry_type == LedgerEntryType.JOB_REFUND]) == 1


def test_late_failure_does_not_return_successful_job_charge(container,user):
    charged_job(container,user)
    with container.store.transaction() as tx:
        job=tx.get(Job,'job_t1'); job.status=JobStatus.SUCCEEDED; tx.put(job,job.job_id)
    completed=container.funding.fail_and_refund('job_t1',error_code=ErrorCode.BACKEND_FAILED,now=NOW+1)
    assert completed.status == JobStatus.SUCCEEDED and completed.balance_returned_cents == 0
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 970
