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
    max_concurrent_jobs: int = 0  # 0 leaves concurrency to the backend.
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
    engine_backend: str = "disabled"
    youtube_download_mode: str = "direct"
    youtube_worker_tokens: tuple[str, ...] = field(default=(), repr=False)
    docker_command: str = "docker"
    engine_container: str = "videotranslator-api-1"
    pricing: PricingConfig = field(default_factory=lambda: PricingConfig(
        pricing_version="usd-cent-v1", point_units_per_minute=10, minimum_point_units=1))
    auth_mode: str = "firebase"
    firebase_web_config: dict = field(default_factory=dict)
    payment_mode: str = "disabled"
    public_app_url: str = "http://localhost:8090"
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
        ("points_1_v1", "USD", 100, 100, "topup-v1"),
        ("points_10_v2", "USD", 1000, 1100, "topup-bonus-v2"),
        ("points_50_v2", "USD", 5000, 6000, "topup-bonus-v2"),
        ("points_100_v2", "USD", 10000, 13000, "topup-bonus-v2"),
    )

    @property
    def processing_available(self) -> bool:
        # Both local profiles still use simulated heavy models. Only the test
        # profile may exercise charging with those models.
        if self.profile == "local-ui":
            return False
        if self.profile == "local-full":
            return self.engine_backend == "docker"
        return True


def _bool(env: str, default: bool) -> bool:
    raw = os.environ.get(env)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def settings_from_env() -> Settings:
    from .payment_environment import profiled, select_environment
    base = Settings()
    download_mode = os.environ.get("YOUTUBE_DOWNLOAD_MODE", "direct")
    worker_tokens = tuple(t.strip() for t in os.environ.get("YOUTUBE_WORKER_TOKENS", "").split(",") if t.strip())
    if download_mode not in {"direct", "home-worker"}:
        raise ValueError("YOUTUBE_DOWNLOAD_MODE must be direct or home-worker")
    if download_mode == "home-worker" and (not worker_tokens or any(len(t) < 32 for t in worker_tokens)):
        raise ValueError("home-worker requires YOUTUBE_WORKER_TOKENS with at least 32 characters per token")
    mode = os.environ.get('PAYMENT_MODE', 'disabled')
    if mode not in {'disabled', 'sandbox', 'live'}:
        raise ValueError('PAYMENT_MODE must be disabled, sandbox or live')
    payment = select_environment(os.environ) if mode != 'disabled' and profiled(os.environ) else None
    pricing = PricingConfig(
        pricing_version=os.environ.get("PRICING_VERSION", base.pricing.pricing_version),
        point_units_per_minute=int(os.environ.get("POINT_UNITS_PER_MINUTE", base.pricing.point_units_per_minute)),
        minimum_point_units=int(os.environ.get("MINIMUM_POINT_UNITS", base.pricing.minimum_point_units)),
    )
    cost = CostPolicy(
        gpu_starts_enabled=_bool("GPU_STARTS_ENABLED", True),
        max_concurrent_jobs=int(os.environ.get("MAX_ACTIVE_GPU_JOBS",
            "1" if os.environ.get("APP_PROFILE") == "azure-jp-t4" else "0")),
        enforce_budgets=_bool("COST_BUDGETS_ENABLED", True),
        hourly_rate_minor=int(os.environ.get("GPU_HOURLY_RATE_MINOR", "75")),
        daily_budget_minor=int(os.environ.get("APP_GPU_DAILY_BUDGET_MINOR", "20000")),
        monthly_budget_minor=int(os.environ.get("APP_GPU_MONTHLY_BUDGET_MINOR", "200000")),
        sku_region_price_version=os.environ.get("GPU_PRICE_VERSION", "local-dev"),
    )
    return Settings(
        youtube_download_mode=download_mode,
        youtube_worker_tokens=worker_tokens,
        engine_backend=os.environ.get("ENGINE_BACKEND", "disabled"),
        docker_command=os.environ.get("DOCKER_COMMAND", "docker"),
        engine_container=os.environ.get("ENGINE_CONTAINER", "videotranslator-api-1"),
        auth_mode=os.environ.get("AUTH_MODE", "firebase"),
        firebase_web_config={key: os.environ.get(env, "") for key, env in {
            "apiKey": "FIREBASE_API_KEY", "authDomain": "FIREBASE_AUTH_DOMAIN",
            "projectId": "FIREBASE_PROJECT_ID", "appId": "FIREBASE_APP_ID"}.items()},
        payment_mode=os.environ.get("PAYMENT_MODE", "disabled"),
        public_app_url=os.environ.get("PUBLIC_APP_URL", base.public_app_url).rstrip("/"),
        profile=os.environ.get("APP_PROFILE", "local-ui"),
        store_path=payment.store_path if payment else os.environ.get("STORE_PATH", "./vt-data/store.db"),
        local_storage_dir=os.environ.get("LOCAL_STORAGE_DIR", base.local_storage_dir),
        local_queue_db=payment.queue_path if payment else os.environ.get("LOCAL_QUEUE_DB", base.local_queue_db),
        pricing=pricing,
        cost=cost,
    )
