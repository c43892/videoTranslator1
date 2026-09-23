"""Fail closed until a real translation worker is connected."""
from ..domain.enums import BackendError, FailureClass
from ..domain.models import BackendStatus


class UnavailableJobBackend:
    name = "unavailable"

    def submit(self, spec, idempotency_key):
        raise BackendError("Translation worker is not connected", failure_class=FailureClass.PERMANENT)

    def get_status(self, backend_job_id):
        return BackendStatus(state="failed", error_message="Translation worker is not connected")

    def cancel(self, backend_job_id):
        return True
