"""Composition root: wires settings + adapters + services per profile (§13.1).

This is the ONLY module allowed to know which concrete adapter sits behind
each port for a given profile; everything else depends on protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from .adapters.conversation import DeepSeekConversationInterpreter, GuidedInterpreter, youtube_url
from .adapters.youtube import YtDlpVideoImporter
from .application.conversations import ConversationService

from .adapters.auth import FirebaseEmulatorIdentityVerifier
from .adapters.fake import (
    FakeIdentityVerifier,
    FakeJobBackend,
    FakeMediaInspectionBackend,
    FakeObjectStorage,
    FakePaymentGateway,
)
from .adapters.ffmpeg import FFmpegMediaInspector
from .adapters.local_jobs import LocalCpuInspectionBackend
from .adapters.local_storage import LocalObjectStorage
from .adapters.unavailable import UnavailableJobBackend
from .application.billing import BillingService, StaticTopUpPricing
from .application.dispatch import OutboxDispatcher, PaymentInboxProcessor
from .application.jobs import JobService
from .application.reconcile import JobReconciler
from .application.uow import BillingUnitOfWork, JobFundingUnitOfWork
from .config import Settings, settings_from_env
from .docstore import MemoryStore, SQLiteStore, Store
from .domain.models import PricingConfig, now_ms


@dataclass
class Container:
    settings: Settings
    store: Store
    identity: object
    storage: object
    job_backend: object
    inspection_backend: object
    gateways: dict
    funding: JobFundingUnitOfWork
    billing_uow: BillingUnitOfWork
    jobs: JobService
    billing: BillingService
    dispatcher: OutboxDispatcher
    inbox: PaymentInboxProcessor
    reconciler: JobReconciler
    conversations: ConversationService

    def seed(self) -> None:
        """Idempotent baseline rows: active pricing config."""
        with self.store.transaction() as tx:
            if tx.get(PricingConfig, self.settings.pricing.pricing_version) is None:
                tx.insert(self.settings.pricing, self.settings.pricing.pricing_version)


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or settings_from_env()
    profile = settings.profile

    store: Store = MemoryStore() if settings.store_path == ":memory:" else SQLiteStore(settings.store_path)

    if profile == "test":
        identity = FakeIdentityVerifier()
        storage = FakeObjectStorage(settings.local_storage_dir)
        job_backend = FakeJobBackend(storage)
        inspection_backend = FakeMediaInspectionBackend()
        gateways = {"stripe": FakePaymentGateway("stripe"), "paypal": FakePaymentGateway("paypal")}
    elif profile in ("local-ui", "local-full"):
        identity = FirebaseEmulatorIdentityVerifier()
        storage = LocalObjectStorage(settings.local_storage_dir, secret=os.environ.get('LOCAL_STORAGE_SIGNING_SECRET'))
        if profile == "local-full":
            if settings.engine_backend == "docker":
                from .adapters.docker_engine import DockerEngineBackend
                job_backend = DockerEngineBackend(storage, docker=settings.docker_command,
                                                 container=settings.engine_container)
                job_backend.check_ready()
            else:
                job_backend = UnavailableJobBackend()
            inspection_backend = LocalCpuInspectionBackend(settings.local_queue_db)
        else:
            job_backend = UnavailableJobBackend()
            inspection_backend = FakeMediaInspectionBackend()
        gateways = {"stripe": FakePaymentGateway("stripe"), "paypal": FakePaymentGateway("paypal")}
    elif profile.startswith("azure"):
        identity = None  # Initialized below according to the explicit auth mode.
        storage = _azure_storage(settings)
        job_backend = _azure_ml_backend(settings)
        inspection_backend = _container_apps_inspection(settings)
        gateways = {}
    else:
        raise ValueError(f"unknown APP_PROFILE {profile!r}")

    if profile != "test":
        if settings.payment_mode in {"sandbox", "live"} and settings.auth_mode == "firebase" and settings.store_path == ":memory:":
            raise ValueError("Payments require a persistent STORE_PATH")
        if settings.auth_mode == "firebase":
            from .adapters.auth import LazyFirebaseIdentityVerifier
            identity = LazyFirebaseIdentityVerifier()
        elif settings.auth_mode != "demo" or profile.startswith("azure"):
            raise ValueError("AUTH_MODE must be firebase (demo is local only)")
        gateways = _live_gateways(settings.payment_mode) if settings.auth_mode == "firebase" and settings.payment_mode in {"sandbox", "live"} else {}

    pricing = settings.pricing
    funding = JobFundingUnitOfWork(
        store, pricing, settings.cost, max_active_jobs_per_user=settings.max_active_jobs_per_user,
        processing_available=settings.processing_available,
    )
    billing_uow = BillingUnitOfWork(store)
    inspector = FFmpegMediaInspector()
    jobs = JobService(store, storage, funding, settings, inspector=inspector)
    billing = BillingService(store, StaticTopUpPricing(settings.topup_packages), gateways)
    dispatcher = OutboxDispatcher(
        store,
        job_backend,
        inspection_backend,
        funding,
        retry_window_ms=settings.retry_window_ms,
        max_retry_attempts=settings.max_retry_attempts,
        lease_ms=settings.dispatcher_lease_ms,
    )
    inbox = PaymentInboxProcessor(store, billing_uow, lease_ms=settings.dispatcher_lease_ms)
    reconciler = JobReconciler(
        store,
        job_backend,
        inspection_backend,
        storage,
        funding,
        settings.cost,
        retry_window_ms=settings.retry_window_ms,
    )
    return Container(
        settings=settings,
        store=store,
        identity=identity,
        storage=storage,
        job_backend=job_backend,
        inspection_backend=inspection_backend,
        gateways=gateways,
        funding=funding,
        billing_uow=billing_uow,
        jobs=jobs,
        billing=billing,
        dispatcher=dispatcher,
        inbox=inbox,
        reconciler=reconciler,
        conversations=ConversationService(store, jobs, storage, settings,
            DeepSeekConversationInterpreter(os.environ["DEEPSEEK_API_KEY"],
                os.environ.get("CHAT_MODEL", "deepseek-chat"),
                os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
            if os.environ.get("DEEPSEEK_API_KEY") and profile != "test" else GuidedInterpreter(),
            YtDlpVideoImporter(), youtube_url, inspector, funding),
    )


def _azure_storage(settings: Settings):  # pragma: no cover - cloud only
    from .adapters.azure_blob import AzureBlobStorage

    return AzureBlobStorage.from_env()


def _azure_ml_backend(settings: Settings):  # pragma: no cover - cloud only
    from .adapters.azure_ml import AzureMLJobBackend

    return AzureMLJobBackend.from_env()


def _container_apps_inspection(settings: Settings):  # pragma: no cover - cloud only
    from .adapters.azure_containerapps import ContainerAppsInspectionBackend

    return ContainerAppsInspectionBackend.from_env()


def _live_gateways(mode="sandbox"):  # pragma: no cover - external credentials
    import os

    from .adapters.payments import PayPalPaymentGateway, StripePaymentGateway

    gateways = {}
    if os.environ.get("STRIPE_SECRET_KEY") and os.environ.get("STRIPE_WEBHOOK_SECRET"):
        if mode == "sandbox" and not os.environ["STRIPE_SECRET_KEY"].startswith(("sk_test_", "rk_test_")):
            raise ValueError("PAYMENT_MODE=sandbox requires a Stripe test key")
        if mode == "live" and not os.environ["STRIPE_SECRET_KEY"].startswith(("sk_live_", "rk_live_")):
            raise ValueError("PAYMENT_MODE=live requires a Stripe live key")
        gateways["stripe"] = StripePaymentGateway(
            os.environ["STRIPE_SECRET_KEY"], os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        )
    if os.environ.get("PAYPAL_CLIENT_ID") and os.environ.get("PAYPAL_CLIENT_SECRET") and os.environ.get("PAYPAL_WEBHOOK_ID"):
        gateways["paypal"] = PayPalPaymentGateway(
            os.environ["PAYPAL_CLIENT_ID"], os.environ["PAYPAL_CLIENT_SECRET"],
            base_url="https://api-m.paypal.com" if mode == "live" else "https://api-m.sandbox.paypal.com",
            webhook_id=os.environ.get("PAYPAL_WEBHOOK_ID", ""),
        )
    return gateways
