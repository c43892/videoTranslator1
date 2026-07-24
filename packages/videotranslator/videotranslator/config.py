"""Runtime settings: env-driven, profile-aware, no secrets in code (§13.1, §14.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .domain.models import PricingConfig

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


@dataclass(frozen=True)
class CostPolicy:
    """Application-layer cost guardrails (§14.4); fail closed when disabled."""

    gpu_starts_enabled: bool = True
    enforce_budgets: bool = True
    hourly_rate_minor: int = 75
    daily_budget_minor: int = 20_000
    monthly_budget_minor: int = 200_000
    sku_region_price_version: str = "local-dev"
    capacity_block_gpu_seconds: int = 14_400  # 4h backlog → awaiting_capacity
    capacity_warn_gpu_seconds: int = 7_200
    runtime_ratio: float = 2.0
    provisioning_allowance_s: int = 900


@dataclass(frozen=True)
class Settings:
    profile: str = "test"
    store_path: str = ":memory:"  # SQLite file for local profiles
    local_storage_dir: str = "./vt-data/objects"
    local_queue_db: str = "./vt-data/scheduler.db"
    pricing: PricingConfig = field(default_factory=PricingConfig)
    cost: CostPolicy = field(default_factory=CostPolicy)
    max_active_jobs_per_user: int = 1
    max_retry_attempts: int = 3
    retry_window_ms: int = 72 * HOUR_MS
    awaiting_input_retention_ms: int = 72 * HOUR_MS
    output_retention_ms: int = 7 * DAY_MS
    max_upload_bytes: int = 2 * 1024**3
    sync_inspect_max_bytes: int = 256 * 1024**2
    sas_ttl_seconds: int = 900
    dispatcher_lease_ms: int = 60_000
    topup_packages: tuple = (
        # package_id, currency, amount_minor, point_units, pricing_version
        ("points_10_v1", "USD", 1000, 1000, "topup-v1"),
        ("points_50_v1", "USD", 5000, 5000, "topup-v1"),
        ("points_100_v1", "USD", 10000, 10000, "topup-v1"),
    )


def _bool(env: str, default: bool) -> bool:
    raw = os.environ.get(env)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def settings_from_env() -> Settings:
    base = Settings()
    pricing = PricingConfig(
        pricing_version=os.environ.get("PRICING_VERSION", base.pricing.pricing_version),
        point_units_per_minute=int(os.environ.get("POINT_UNITS_PER_MINUTE", base.pricing.point_units_per_minute)),
        minimum_point_units=int(os.environ.get("MINIMUM_POINT_UNITS", base.pricing.minimum_point_units)),
    )
    cost = CostPolicy(
        gpu_starts_enabled=_bool("GPU_STARTS_ENABLED", True),
        enforce_budgets=_bool("COST_BUDGETS_ENABLED", True),
        hourly_rate_minor=int(os.environ.get("GPU_HOURLY_RATE_MINOR", "75")),
        daily_budget_minor=int(os.environ.get("APP_GPU_DAILY_BUDGET_MINOR", "20000")),
        monthly_budget_minor=int(os.environ.get("APP_GPU_MONTHLY_BUDGET_MINOR", "200000")),
        sku_region_price_version=os.environ.get("GPU_PRICE_VERSION", "local-dev"),
    )
    return Settings(
        profile=os.environ.get("APP_PROFILE", "test"),
        store_path=os.environ.get("STORE_PATH", base.store_path),
        local_storage_dir=os.environ.get("LOCAL_STORAGE_DIR", base.local_storage_dir),
        local_queue_db=os.environ.get("LOCAL_QUEUE_DB", base.local_queue_db),
        pricing=pricing,
        cost=cost,
    )
