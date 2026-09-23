"""Azure ML job backend (§13.4) and the deterministic name mapping (§13.4).

SDK imported lazily; the name mapping is pure and unit-tested.
"""

from __future__ import annotations

import os
import re
from hashlib import sha256

from ..domain.models import BackendJobRef, BackendStatus, InspectionSpec, JobSpec, MediaInspectionResult


def azure_job_name(outbox_id: str, prefix: str = "vt") -> str:
    """§13.4: deterministic, legal, unique; full SHA-256 keeps uniqueness."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", outbox_id).strip("-_").lower()
    digest = sha256(outbox_id.encode("utf-8")).hexdigest()
    return f"{prefix}-{normalized[:180]}-{digest}"


_STATE_MAP = {
    "Queued": "queued",
    "Preparing": "provisioning",
    "Starting": "provisioning",
    "Provisioning": "provisioning",
    "Running": "running",
    "Completed": "succeeded",
    "Failed": "failed",
    "Canceled": "cancelled",
    "CancelRequested": "running",
    "NotResponding": "running",
}


class AzureMLJobBackend:  # pragma: no cover - cloud only
    name = "azureml"

    def __init__(self, subscription_id: str, resource_group: str, workspace: str):
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        self._ml = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

    @classmethod
    def from_env(cls) -> "AzureMLJobBackend":
        return cls(
            os.environ["AZURE_SUBSCRIPTION_ID"],
            os.environ["AZURE_RESOURCE_GROUP"],
            os.environ["AZURE_ML_WORKSPACE"],
        )

    def submit(self, spec: JobSpec, idempotency_key: str) -> BackendJobRef:
        from azure.ai.ml import command

        name = azure_job_name(idempotency_key, "vt")
        # Get-or-create: the name is derived from the idempotency key, so a
        # retry or lease takeover must adopt the existing job, never make a
        # second one (§16.1).
        try:
            existing = self._ml.jobs.get(name)
        except Exception:
            existing = None
        if existing is not None:
            return BackendJobRef(existing.name, name)
        job = command(
            name=name,
            command=f"python -m videotranslator.worker.cli /specs/{name}.json",
            environment=os.environ["AZURE_ML_ENVIRONMENT"],
            compute=os.environ["AZURE_ML_COMPUTE"],
            limits={"timeout": spec.max_runtime_seconds},
        )
        created = self._ml.jobs.create_or_update(job)
        return BackendJobRef(created.name, name)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        try:
            job = self._ml.jobs.get(backend_job_id)
        except Exception:
            return BackendStatus(state="not_found")
        return BackendStatus(state=_STATE_MAP.get(job.status, "running"), error_message=getattr(job, "error", None))

    def cancel(self, backend_job_id: str) -> bool:
        try:
            self._ml.jobs.cancel(backend_job_id)
            return True
        except Exception:
            return False
