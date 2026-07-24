"""Retry/backoff policy shared by Outbox dispatchers and the Inbox processor (§16.2)."""

from __future__ import annotations

import random

from .enums import BackendError, FailureClass


def backoff_ms(attempt: int, base_s: int, cap_s: int, jitter_frac: float = 0.2) -> int:
    """min(base × 2^(attempt-1), cap) seconds with ±jitter, in milliseconds."""
    delay = min(base_s * (2 ** max(0, attempt - 1)), cap_s)
    if jitter_frac:
        delay = delay * (1 + random.uniform(-jitter_frac, jitter_frac))
    return int(delay * 1000)


def classify(exc: Exception) -> FailureClass:
    if isinstance(exc, BackendError):
        return exc.failure_class
    return FailureClass.RETRYABLE
