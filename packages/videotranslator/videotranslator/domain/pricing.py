"""Pricing and duration math (§4.1–4.3).

Durations travel as integer milliseconds; the ffprobe raw string is converted
via ``Decimal`` exactly once and never round-trips through a binary float.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_CEILING

from .enums import DomainError, ErrorCode
from .models import JobQuote, MediaType, PricingConfig

MEDIA_HARD_LIMIT_MS = 1_800_000  # 30 minutes (§4.5)


def probe_duration_to_ms(raw: str) -> int:
    """ffprobe duration string → integer ms, Decimal ceiling, never a float."""
    try:
        seconds = Decimal(raw.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise DomainError(f"unparseable ffprobe duration: {raw!r}", code=ErrorCode.MEDIA_INSPECTION_FAILED) from exc
    if seconds <= 0:
        raise DomainError(f"non-positive duration: {raw!r}", code=ErrorCode.MEDIA_INSPECTION_FAILED)
    return int((seconds * 1000).to_integral_value(rounding=ROUND_CEILING))


def quote_job(config: PricingConfig, duration_ms: int, media_type: MediaType) -> JobQuote:
    """§4.2: single ceiling at the final Point Unit boundary."""
    if duration_ms <= 0:
        raise DomainError("duration must be positive", code=ErrorCode.MEDIA_INSPECTION_FAILED)
    if duration_ms > MEDIA_HARD_LIMIT_MS:
        raise DomainError(
            f"media exceeds the 30-minute limit ({duration_ms} ms)", code=ErrorCode.MEDIA_TOO_LONG
        )
    units = math.ceil(duration_ms * config.point_units_per_minute / 60_000)
    units = max(units, config.minimum_point_units)
    return JobQuote(duration_ms=duration_ms, point_units=units, pricing_version=config.pricing_version)


def estimated_gpu_seconds(duration_ms: int, runtime_ratio: float, provisioning_allowance_s: int) -> int:
    """§13.6 admission estimate, calibrated from real runs."""
    return math.ceil(duration_ms / 1000 * runtime_ratio) + provisioning_allowance_s


def max_runtime_seconds(duration_ms: int) -> int:
    """§14.4 per-job hard run cap, also written to the backend timeout."""
    return min(14_400, max(1_800, math.ceil(duration_ms / 1000 * 4 + 900)))
