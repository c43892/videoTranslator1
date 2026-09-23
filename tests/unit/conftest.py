"""Shared fixtures: a test container with memory store and fake adapters."""

from __future__ import annotations

import pytest

from videotranslator.application.uow import CompleteInspectionCommand
from videotranslator.bootstrap import build_container
from videotranslator.config import CostPolicy, Settings
from videotranslator.domain.enums import JobStatus
from videotranslator.domain.models import CapacityCounter, User, PricingConfig

NOW = 1_800_000_000_000  # fixed epoch ms for deterministic tests


@pytest.fixture()
def container(tmp_path):
    settings = Settings(
        profile="test",
        store_path=":memory:",
        local_storage_dir=str(tmp_path / "objects"),
        cost=CostPolicy(daily_budget_minor=10_000, monthly_budget_minor=100_000),
        # Existing ledger regressions use an explicit historical 100-unit rate.
        pricing=PricingConfig(),
    )
    c = build_container(settings)
    c.seed()
    with c.store.transaction() as tx:
        tx.insert(CapacityCounter(counter_id="global"), "global")
    return c


@pytest.fixture()
def user(container):
    with container.store.transaction() as tx:
        u = User(user_id="u1", email="u1@example.test", point_balance_units=0, created_at=NOW, updated_at=NOW)
        tx.insert(u, "u1")
    return u


def give_balance(container, user_id: str, units: int, now: int = NOW) -> None:
    with container.store.transaction() as tx:
        u = tx.get(User, user_id)
        u.point_balance_units = units
        tx.put(u, user_id)


def make_job_via_inspection(container, user_id: str = "u1", *, duration_ms: int = 60_000, job_id: str = "job_t1", now: int = NOW):
    """Drive the canonical entry: complete_inspection with a trusted result."""
    return container.funding.complete_inspection(
        CompleteInspectionCommand(
            job_id=job_id,
            owner_user_id=user_id,
            original_filename="clip.mp4",
            media_type="video",
            target_language="English",
            input_object_key=f"users/{user_id}/jobs/{job_id}/input.mp4",
            duration_ms=duration_ms,
            duration_probe_raw=f"{duration_ms / 1000:.6f}",
            inspection_attempt=1,
            now=now,
        )
    )


def job_status(container, job_id: str) -> JobStatus:
    with container.store.transaction() as tx:
        from videotranslator.domain.models import Job

        return tx.get(Job, job_id).status
