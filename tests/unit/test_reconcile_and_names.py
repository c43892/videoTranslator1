"""§18.1: reconciler sync, azure job name mapping, end-to-end happy path."""

from __future__ import annotations

import re

from videotranslator.adapters.azure_ml import azure_job_name
from videotranslator.domain.enums import JobStatus
from videotranslator.domain.models import CapacityCounter, User

from .conftest import NOW, give_balance, job_status, make_job_via_inspection


class TestAzureJobName:
    def test_legal_chars_and_length(self):
        name = azure_job_name("job-submit:job_01abc:1")
        assert re.fullmatch(r"[a-z][a-z0-9_-]*", name)
        assert len(name) <= 255
        assert name.startswith("vt-")

    def test_deterministic_and_unique_after_normalization(self):
        a = azure_job_name("job-submit:job_01abc:1")
        b = azure_job_name("job-submit:job_01abc:1")
        c = azure_job_name("job-submit/job_01abc/1")  # normalizes to the same string
        assert a == b
        assert a != c  # full SHA-256 keeps them distinct

    def test_prefixes(self):
        assert azure_job_name("inspection:job_x:1", "vti").startswith("vti-")


class TestReconciler:
    def test_full_happy_path_to_succeeded(self, container, user):
        give_balance(container, "u1", 500)
        make_job_via_inspection(container)
        container.storage.put_bytes("users/u1/jobs/job_t1/input.mp4", b"x")
        container.dispatcher.dispatch_due_jobs(now=NOW)
        assert job_status(container, "job_t1") == JobStatus.PROVISIONING

        container.reconciler.reconcile_once(now=NOW + 500)   # backend: provisioning
        container.reconciler.reconcile_once(now=NOW + 1_000)  # backend: running
        assert job_status(container, "job_t1") == JobStatus.RUNNING
        container.reconciler.reconcile_once(now=NOW + 2_000)  # backend: succeeded
        status = job_status(container, "job_t1")
        assert status == JobStatus.SUCCEEDED

        with container.store.transaction() as tx:
            from videotranslator.domain.models import Job

            job = tx.get(Job, "job_t1")
            assert job.output_object_key == "users/u1/jobs/job_t1/result.mp4"
            assert job.progress_percent == 100
            assert job.output_expires_at is not None
            counter = tx.get(CapacityCounter, "global")
            assert counter.reserved_gpu_seconds == 0  # fully released

    def test_backend_failure_refunds_and_marks_retryable(self, container, user):
        from videotranslator.adapters.fake import FakeJobBackend

        give_balance(container, "u1", 500)
        container.job_backend._fail_on_poll = 2
        container.dispatcher._job_backend = container.job_backend
        container.reconciler._job_backend = container.job_backend
        make_job_via_inspection(container)
        container.dispatcher.dispatch_due_jobs(now=NOW)
        container.reconciler.reconcile_once(now=NOW + 1_000)
        container.reconciler.reconcile_once(now=NOW + 2_000)
        assert job_status(container, "job_t1") == JobStatus.FAILED
        with container.store.transaction() as tx:
            assert tx.get(User, "u1").point_balance_units == 500

    def test_async_inspection_closes_to_queued(self, container, user):
        give_balance(container, "u1", 500)
        ticket = container.jobs.create_upload(
            "u1", filename="talk.mp4", size_bytes=3, target_language="English", now=NOW
        )
        container.storage.put_bytes(ticket.object_key, b"abc")
        job = container.jobs.complete_upload(ticket.upload_id, "u1", now=NOW)
        assert job.status == JobStatus.INSPECTING

        assert container.dispatcher.dispatch_due_inspections(now=NOW) == 1
        container.reconciler.reconcile_once(now=NOW + 500)   # inspection running
        assert job_status(container, job.job_id) == JobStatus.INSPECTING
        container.reconciler.reconcile_once(now=NOW + 1_000)  # inspection done → handler
        assert job_status(container, job.job_id) == JobStatus.QUEUED
        with container.store.transaction() as tx:
            u = tx.get(User, "u1")
            assert u.point_balance_units == 400  # fake inspection says 60s → 100 units
