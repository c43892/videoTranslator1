"""§18.1: payment idempotency, dispatcher/cancel race, retry cap, stale versions."""

from __future__ import annotations

import threading

import pytest

from videotranslator.domain.enums import (
    DomainError,
    ErrorCode,
    JobStatus,
    OutboxStatus,
    PaymentEventStatus,
    StaleVersion,
)
from videotranslator.domain.models import Job, JobOutbox, LedgerEntry, Payment, PaymentEvent, User

from .conftest import NOW, give_balance, job_status, make_job_via_inspection


class TestPayments:
    def _buy(self, container, user_id="u1"):
        session = container.billing.create_session(user_id, package_id="points_10_v1", provider="stripe", now=NOW)
        gateway = container.gateways["stripe"]
        headers, body = gateway.make_webhook(session.payment_id)
        key = container.billing.record_webhook("stripe", headers, body)
        return session.payment_id, key

    def test_webhook_posts_points_once(self, container, user):
        _, key = self._buy(container)
        container.inbox.process_due(now=NOW)
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 1000
        container.inbox.process_due(now=NOW)  # duplicate delivery absorbed
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 1000
            purchases = [e for e in tx.query(LedgerEntry, where=("user_id", "==", "u1"))]
            assert len(purchases) == 1

    def test_duplicate_webhook_ingest_is_absorbed(self, container, user):
        _, key1 = self._buy(container)
        gateway = container.gateways["stripe"]
        with container.store.transaction() as tx:
            payment = tx.query(Payment, where=("user_id", "==", "u1"))[0]
        headers, body = gateway.make_webhook(payment.payment_id)
        key2 = container.billing.record_webhook("stripe", headers, body)
        assert key1 == key2
        with container.store.transaction() as tx:
            events = tx.query(PaymentEvent)
            assert len(events) == 1

    def test_reversal_allows_negative_and_holds_billing(self, container, user):
        payment_id, _ = self._buy(container)
        container.inbox.process_due(now=NOW)  # +1000 → balance 1000
        # spend 100 on a job so the reversal below drives the balance negative
        make_job_via_inspection(container, job_id="job_neg")  # -100 → 900
        container.billing_uow.reverse_payment(payment_id, now=NOW)  # -1000 → -100
        with container.store.transaction() as tx:
            u = tx.get(User, "u1")
            assert u.point_balance_units == -100
            assert u.billing_status == "hold"
        # a later purchase first offsets the negative balance, then clears hold
        payment_id2, _ = self._buy(container)
        container.inbox.process_due(now=NOW)  # +1000 → 900
        with container.store.transaction() as tx:
            u = tx.get(User, "u1")
            assert u.point_balance_units == 900
            assert u.billing_status == "clear"

    def test_paypal_capture_posts_points(self, container, user):
        session = container.billing.create_session("u1", package_id="points_10_v1", provider="paypal", now=NOW)
        key = container.billing.capture_paypal(session.provider_order_id, "u1", now=NOW)
        assert key.startswith("paypal:capture:")
        container.inbox.process_due(now=NOW)
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 1000
        # replay via a later webhook for the same capture must not double-post
        container.inbox.process_due(now=NOW)
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 1000


class TestDispatcherCancelRace:
    def test_cancel_during_claim_leaves_exactly_one_outcome(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        results = []

        def dispatch():
            results.append(("dispatch", container.dispatcher.dispatch_due_jobs(now=NOW)))

        def cancel():
            results.append(("cancel", container.funding.cancel("job_t1", now=NOW).outcome))

        threads = [threading.Thread(target=dispatch), threading.Thread(target=cancel)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        status = job_status(container, "job_t1")
        assert status in (JobStatus.PROVISIONING, JobStatus.CANCELLED, JobStatus.CANCELLING)
        with container.store.transaction() as tx:
            outbox = tx.get(JobOutbox, "job-submit:job_t1:1")
            job = tx.get(Job, "job_t1")
            if job.status == JobStatus.CANCELLED:
                assert outbox.status == OutboxStatus.CANCELLED
                assert tx.get(User, "u1").point_balance_units == 500  # refunded
            else:
                assert outbox.status == OutboxStatus.COMPLETED

    def test_dispatcher_submits_and_moves_to_provisioning(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        assert container.dispatcher.dispatch_due_jobs(now=NOW) == 1
        assert job_status(container, "job_t1") == JobStatus.PROVISIONING
        with container.store.transaction() as tx:
            outbox = tx.get(JobOutbox, "job-submit:job_t1:1")
            assert outbox.status == OutboxStatus.COMPLETED
            assert outbox.backend_job_id is not None

    def test_submit_failure_retries_then_dead_letters(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)

        class FailingBackend:
            name = "fake"

            def submit(self, spec, idempotency_key):
                from videotranslator.domain.enums import BackendError, FailureClass

                raise BackendError("boom", failure_class=FailureClass.PERMANENT)

            def get_status(self, backend_job_id):
                from videotranslator.domain.models import BackendStatus

                return BackendStatus(state="not_found")

            def cancel(self, backend_job_id):
                return True

        container.dispatcher._job_backend = FailingBackend()
        container.dispatcher.dispatch_due_jobs(now=NOW)
        assert job_status(container, "job_t1") == JobStatus.FAILED
        with container.store.transaction() as tx:
            outbox = tx.get(JobOutbox, "job-submit:job_t1:1")
            assert outbox.status == OutboxStatus.DEAD_LETTER
            assert tx.get(User, "u1").point_balance_units == 500  # full refund on dead letter
            job = tx.get(Job, "job_t1")
            assert job.retry_allowed is True


class TestRetry:
    def _fail_running(self, container, job_id: str, now: int = NOW):
        """Drive a charged job to provisioning, then fail it as a backend failure."""
        container.dispatcher.dispatch_due_jobs(now=now)
        container.funding.fail_and_refund(
            job_id, error_code=ErrorCode.BACKEND_FAILED, refund=True,
            retry_allowed=True, retry_window_ms=72 * 3_600_000, now=now,
        )

    def test_retry_creates_new_job_and_charges_again(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        self._fail_running(container, "job_t1")
        # storage.exists check: put the input object
        container.storage.put_bytes("users/u1/jobs/job_t1/input.mp4", b"x")
        child = container.jobs.retry("job_t1", "u1", now=NOW)
        assert child.retry_of_job_id == "job_t1"
        assert child.attempt_number == 2
        assert child.status == JobStatus.QUEUED
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 400  # 500 - 100 (child charge)
        again = container.jobs.retry("job_t1", "u1", now=NOW)  # same idempotency key
        assert again.job_id == child.job_id

    def test_retry_exhausted_after_cap(self, container, user):
        give_balance(container, "u1", 100_000)
        container.storage.put_bytes("users/u1/jobs/job_t1/input.mp4", b"x")
        make_job_via_inspection(container)
        parent = "job_t1"
        for attempt in (2, 3):
            self._fail_running(container, parent)
            child = container.jobs.retry(parent, "u1", now=NOW)
            parent = child.job_id
        self._fail_running(container, parent)
        with pytest.raises(DomainError) as exc:
            container.jobs.retry(parent, "u1", now=NOW)
        assert exc.value.code == ErrorCode.RETRY_EXHAUSTED


class TestStaleVersion:
    def test_language_patch_conflict_rejected(self, container, user):
        make_job_via_inspection(container)
        job = container.jobs.patch_language("job_t1", "u1", language="French", expected_status_version=1)
        assert job.target_language == "French"
        with pytest.raises(StaleVersion):
            container.jobs.patch_language("job_t1", "u1", language="German", expected_status_version=1)

    def test_capacity_counter_blocks_at_4h_backlog(self, container, user):
        from videotranslator.domain.models import CapacityCounter

        with container.store.transaction() as tx:
            counter = tx.get(CapacityCounter, "global")
            counter.reserved_gpu_seconds = 14_400
            tx.put(counter, "global")
        give_balance(container, "u1", 500)
        result = make_job_via_inspection(container)
        assert result.outcome == "awaiting_capacity"
        assert result.block_reason == "backlog_exceeded"
