"""§18.1: state machine legality, charge atomicity, idempotent completion."""

from __future__ import annotations

import threading

import pytest

from videotranslator.domain.enums import (
    ALLOWED_TRANSITIONS,
    InsufficientCredits,
    InvalidTransition,
    JobStatus,
    check_transition,
)
from videotranslator.domain.models import Job, JobOutbox, LedgerEntry, OutboxStatus, User

from .conftest import NOW, give_balance, job_status, make_job_via_inspection


class TestStateMachine:
    def test_every_documented_edge_is_allowed(self):
        documented = [
            ("uploaded", "inspecting"),
            ("inspecting", "awaiting_credits"),
            ("inspecting", "awaiting_capacity"),
            ("inspecting", "queued"),
            ("inspecting", "failed"),
            ("awaiting_credits", "queued"),
            ("awaiting_credits", "awaiting_capacity"),
            ("awaiting_credits", "expired"),
            ("awaiting_capacity", "queued"),
            ("awaiting_capacity", "awaiting_credits"),
            ("awaiting_capacity", "expired"),
            ("queued", "submitting"),
            ("queued", "cancelled"),
            ("submitting", "provisioning"),
            ("submitting", "failed"),
            ("submitting", "cancelling"),
            ("provisioning", "cancelling"),
            ("provisioning", "running"),
            ("provisioning", "failed"),
            ("cancelling", "cancelled"),
            ("cancelling", "failed"),
            ("cancelling", "running"),
            ("running", "cancelling"),
            ("running", "succeeded"),
            ("running", "failed"),
            ("succeeded", "expired"),
        ]
        for src, dst in documented:
            check_transition(JobStatus(src), JobStatus(dst))

    def test_failed_never_returns_to_queued(self):
        with pytest.raises(InvalidTransition):
            check_transition(JobStatus.FAILED, JobStatus.QUEUED)

    def test_no_self_transitions(self):
        for status in JobStatus:
            assert status not in ALLOWED_TRANSITIONS[status]


class TestCharge:
    def test_insufficient_balance_lands_in_awaiting_credits(self, container, user):
        result = make_job_via_inspection(container)
        assert result.outcome == "awaiting_credits"
        assert job_status(container, "job_t1") == JobStatus.AWAITING_CREDITS

    def test_sufficient_balance_charges_and_queues(self, container, user):
        give_balance(container, "u1", 500)
        result = make_job_via_inspection(container)
        assert result.outcome == "queued"
        assert job_status(container, "job_t1") == JobStatus.QUEUED
        with container.store.transaction() as tx:
            u = tx.get(User, "u1")
            assert u.point_balance_units == 400  # 60s → 100 units charged
            outbox = tx.get(JobOutbox, "job-submit:job_t1:1")
            assert outbox is not None and outbox.status == OutboxStatus.PENDING
            charge = tx.get(LedgerEntry, result.job.charged_ledger_entry_id)
            assert charge.delta_units == -100
            assert charge.balance_after_units == 400

    def test_manual_start_charges_once(self, container, user):
        make_job_via_inspection(container)
        give_balance(container, "u1", 100)
        first = container.funding.fund_and_enqueue("job_t1", now=NOW)
        assert first.outcome == "queued"
        second = container.funding.fund_and_enqueue("job_t1", now=NOW)
        assert second.outcome == "queued"  # idempotent replay, not a second charge
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 0
            charges = tx.query(LedgerEntry, where=("job_id", "==", "job_t1"))
            assert len(charges) == 1

    def test_concurrent_charge_only_one_wins(self, container, user):
        make_job_via_inspection(container)
        give_balance(container, "u1", 100)
        outcomes = []

        def start():
            try:
                outcomes.append(container.funding.fund_and_enqueue("job_t1", now=NOW).outcome)
            except InsufficientCredits:
                outcomes.append("insufficient")

        threads = [threading.Thread(target=start) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count("queued") >= 1
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 0
            assert len(tx.query(LedgerEntry, where=("job_id", "==", "job_t1"))) == 1

    def test_repeated_inspection_completion_quotes_once(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        again = make_job_via_inspection(container)  # replay: job already queued
        assert again.job.quoted_point_units == 100
        with container.store.transaction() as tx:
            assert len(tx.query(LedgerEntry, where=("job_id", "==", "job_t1"))) == 1


class TestCancel:
    def test_queued_cancel_refunds_and_cancels_outbox(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        result = container.funding.cancel("job_t1", now=NOW)
        assert result.outcome == "cancelled_refunded"
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 500
            outbox = tx.get(JobOutbox, "job-submit:job_t1:1")
            assert outbox.status == OutboxStatus.CANCELLED
            refunds = tx.query(LedgerEntry, where=("job_id", "==", "job_t1"))
            assert any(e.delta_units == 100 for e in refunds)
