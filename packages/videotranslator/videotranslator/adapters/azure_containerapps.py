"""Container Apps CPU inspection backend (§13.3); SDK imported lazily."""

from __future__ import annotations

import os

from ..domain.models import BackendJobRef, BackendStatus, InspectionSpec, MediaInspectionResult
from .azure_ml import azure_job_name


class ContainerAppsInspectionBackend:  # pragma: no cover - cloud only
    name = "container-apps-inspection"

    def __init__(self, resource_group: str, job_name: str, storage):
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.appcontainers import ContainerAppsAPIClient

        self._rg = resource_group
        self._job_name = job_name
        self._client = ContainerAppsAPIClient(DefaultAzureCredential(), os.environ["AZURE_SUBSCRIPTION_ID"])
        self._storage = storage
        self._executions: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "ContainerAppsInspectionBackend":
        from .azure_blob import AzureBlobStorage

        return cls(
            os.environ["AZURE_RESOURCE_GROUP"],
            os.environ.get("INSPECTION_JOB_NAME", "vt-inspection"),
            AzureBlobStorage.from_env(),
        )

    def submit(self, spec: InspectionSpec, idempotency_key: str) -> BackendJobRef:
        name = azure_job_name(idempotency_key, "vti")[:63]
        execution = self._client.jobs.begin_start(self._rg, self._job_name).result()
        self._executions[idempotency_key] = execution.name
        return BackendJobRef(idempotency_key, execution.name)

    def get_status(self, backend_job_id: str) -> BackendStatus:
        result_key = self._result_key(backend_job_id)
        if self._storage.exists(result_key):
            return BackendStatus(state="succeeded")
        return BackendStatus(state="running")

    def get_result(self, backend_job_id: str) -> MediaInspectionResult:
        import json
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            path = Path(tmp.name)
        self._storage.download(self._result_key(backend_job_id), path)
        return MediaInspectionResult.from_json_dict(json.loads(path.read_text()))

    @staticmethod
    def _result_key(backend_job_id: str) -> str:
        # inspection:<job_id>:<attempt> → users/.../inspection.json comes from the spec
        parts = backend_job_id.split(":")
        job_id = parts[1] if len(parts) >= 2 else backend_job_id
        return f"inspections/{job_id}/result.json"
