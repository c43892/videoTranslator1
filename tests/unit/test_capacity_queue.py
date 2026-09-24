"""Confirmed capacity waits resume automatically, without duplicate debits."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from videotranslator.domain.models import Job, JobOutbox, LedgerEntry, User
from .conftest import NOW, give_balance, make_job_via_inspection


def test_paused_queue_remains_retryable_and_resumes_once(container, user):
    give_balance(container, 'u1', 500)
    enabled = container.funding._cost
    container.funding._cost = replace(enabled, gpu_starts_enabled=False)
    result = make_job_via_inspection(container)
    assert result.job.capacity_wait_reason == 'gpu_starts_disabled'
    for step in range(1, 5):
        container.reconciler.reconcile_once(now=NOW + step * 10_000)
    with container.store.transaction() as tx:
        assert tx.get(Job, 'job_t1').status == 'awaiting_capacity'
        assert tx.get(User, 'u1').point_balance_units == 500
        assert not tx.query(JobOutbox)
    container.funding._cost = enabled
    container.reconciler.reconcile_once(now=NOW + 50_000)
    container.reconciler.reconcile_once(now=NOW + 60_000)
    with container.store.transaction() as tx:
        job = tx.get(Job, 'job_t1')
        assert job.status == 'queued' and not job.capacity_wait_reason
        assert tx.get(User, 'u1').point_balance_units == 400
        assert len(tx.query(LedgerEntry)) == len(tx.query(JobOutbox)) == 1


def test_second_job_automatically_admitted_after_first_finishes(container, user):
    give_balance(container, 'u1', 500)
    make_job_via_inspection(container)
    result = make_job_via_inspection(container, job_id='job_second', now=NOW + 1)
    assert result.outcome == 'awaiting_capacity'
    container.storage.put_bytes('users/u1/jobs/job_t1/input.mp4', b'x')
    container.dispatcher.dispatch_due_jobs(now=NOW)
    for offset in (500, 1000, 2000, 11_000):
        container.reconciler.reconcile_once(now=NOW + offset)
    with container.store.transaction() as tx:
        assert tx.get(Job, 'job_t1').status == 'succeeded'
        assert tx.get(Job, 'job_second').status == 'queued'
        assert tx.get(User, 'u1').point_balance_units == 300


def test_global_single_slot_and_fifo_wait_for_other_users(container, user):
    give_balance(container, 'u1', 500)
    container.funding._cost = replace(container.funding._cost, max_concurrent_jobs=1)
    with container.store.transaction() as tx:
        tx.insert(User(user_id='u2', point_balance_units=500), 'u2')
    make_job_via_inspection(container)
    make_job_via_inspection(container, user_id='u2', job_id='second', now=NOW+1)
    make_job_via_inspection(container, user_id='u2', job_id='third', now=NOW+2)
    container.funding.cancel('job_t1', now=NOW+5000)
    container.reconciler.reconcile_once(now=NOW+20_000)
    with container.store.transaction() as tx:
        assert tx.get(Job,'second').status == 'queued'
        assert tx.get(Job,'third').status == 'awaiting_capacity'
        assert tx.get(User,'u2').point_balance_units == 400


def test_cancelling_waiting_task_prevents_later_charge(container, user):
    give_balance(container, 'u1', 500)
    enabled = container.funding._cost
    container.funding._cost = replace(enabled, gpu_starts_enabled=False)
    make_job_via_inspection(container)
    assert container.jobs.cancel('job_t1','u1',now=NOW+1000).job.status == 'cancelled'
    container.funding._cost = enabled
    container.reconciler.reconcile_once(now=NOW+20_000)
    with container.store.transaction() as tx:
        assert not tx.query(JobOutbox) and not tx.query(LedgerEntry)
        assert tx.get(User,'u1').point_balance_units == 500


def test_concurrent_schedulers_and_manual_start_debit_once(container, user):
    give_balance(container, 'u1', 500)
    enabled = container.funding._cost
    container.funding._cost = replace(enabled, gpu_starts_enabled=False)
    make_job_via_inspection(container)
    container.funding._cost = enabled
    def race(index):
        if index % 2:
            container.funding.fund_and_enqueue('job_t1',now=NOW+20_000)
        else:
            container.reconciler.reconcile_once(now=NOW+20_000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(race,range(8)))
    with container.store.transaction() as tx:
        assert tx.get(User,'u1').point_balance_units == 400
        assert len(tx.query(LedgerEntry)) == len(tx.query(JobOutbox)) == 1


def test_budget_checks_initial_runtime_before_reserving(container, user):
    give_balance(container, 'u1', 500)
    container.funding._cost = replace(container.funding._cost, daily_budget_minor=1, monthly_budget_minor=1)
    result = make_job_via_inspection(container)
    assert result.outcome == 'awaiting_capacity' and result.block_reason == 'cost_budget_exceeded'
    assert result.job.max_runtime_seconds == 1800
    container.reconciler.reconcile_once(now=NOW+20_000)
    with container.store.transaction() as tx:
        assert tx.get(Job,'job_t1').capacity_wait_reason == 'cost_budget_exceeded'
        assert tx.get(User,'u1').point_balance_units == 500
        assert not tx.query(JobOutbox)
