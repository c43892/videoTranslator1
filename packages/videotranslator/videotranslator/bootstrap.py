"""Composition root: wires settings + adapters + services per profile (§13.1).

This is the ONLY module allowed to know which concrete adapter sits behind
each port for a given profile; everything else depends on protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters.auth import FirebaseEmulatorIdentityVerifier, FirebaseIdentityVerifier
from .adapters.fake import (
    FakeIdentityVerifier,
    FakeJobBackend,
    FakeMediaInspectionBackend,
    FakeObjectStorage,
    FakePaymentGateway,
)
from .adapters.ffmpeg import FFmpegMediaInspector
from .adapters.local_jobs import LocalCpuInspectionBackend, LocalJobBackend
from .adapters.local_storage import LocalObjectStorage
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
        storage = LocalObjectStorage(settings.local_storage_dir)
        if profile == "local-full":
            job_backend = LocalJobBackend(settings.local_queue_db)
            inspection_backend = LocalCpuInspectionBackend(settings.local_queue_db)
        else:
            job_backend = FakeJobBackend(storage)
            inspection_backend = FakeMediaInspectionBackend()
        gateways = {"stripe": FakePaymentGateway("stripe"), "paypal": FakePaymentGateway("paypal")}
    elif profile.startswith("azure"):
        identity = FirebaseIdentityVerifier()
        storage = _azure_storage(settings)
        job_backend = _azure_ml_backend(settings)
        inspection_backend = _container_apps_inspection(settings)
        gateways = _live_gateways()
    else:
        raise ValueError(f"unknown APP_PROFILE {profile!r}")

    pricing = settings.pricing
    funding = JobFundingUnitOfWork(
        store, pricing, settings.cost, max_active_jobs_per_user=settings.max_active_jobs_per_user
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


def _live_gateways():  # pragma: no cover - cloud only
    import os

    from .adapters.payments import PayPalPaymentGateway, StripePaymentGateway

    gateways = {}
    if os.environ.get("STRIPE_SECRET_KEY"):
        gateways["stripe"] = StripePaymentGateway(
            os.environ["STRIPE_SECRET_KEY"], os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        )
    if os.environ.get("PAYPAL_CLIENT_ID"):
        gateways["paypal"] = PayPalPaymentGateway(
            os.environ["PAYPAL_CLIENT_ID"], os.environ["PAYPAL_CLIENT_SECRET"]
        )
    return gateways
