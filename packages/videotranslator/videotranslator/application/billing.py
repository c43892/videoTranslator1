"""Billing application service: checkout sessions, webhooks, captures (§10, §12.3)."""

from __future__ import annotations

from dataclasses import dataclass

from ..docstore import Store
from ..domain.enums import DomainError, ErrorCode, Forbidden, NotFound
from ..domain.models import Payment, PaymentEvent, PaymentEventStatus, TopUpPackage, now_ms
from ..ids import new_id
from ..ports import PaymentGateway, PaymentSession, PaymentSessionRequest


@dataclass(frozen=True)
class StaticTopUpPricing:
    """Versioned top-up catalog from Settings (§11.7)."""

    packages: tuple

    def list_packages(self, currency: str) -> list[TopUpPackage]:
        return [
            TopUpPackage(package_id=p[0], currency=p[1], amount_minor=p[2], point_units=p[3], pricing_version=p[4])
            for p in self.packages
            if p[1] == currency
        ]

    def get_package(self, package_id: str) -> TopUpPackage:
        for p in self.packages:
            if p[0] == package_id:
                return TopUpPackage(package_id=p[0], currency=p[1], amount_minor=p[2], point_units=p[3], pricing_version=p[4])
        raise NotFound(f"package {package_id}")


class BillingService:
    def __init__(self, store: Store, pricing: StaticTopUpPricing, gateways: dict[str, PaymentGateway]):
        self._store = store
        self._pricing = pricing
        self._gateways = gateways

    def list_packages(self, currency: str = "USD") -> list[TopUpPackage]:
        return self._pricing.list_packages(currency)

    def create_session(
        self,
        user_id: str,
        *,
        package_id: str,
        provider: str,
        success_url: str = "",
        cancel_url: str = "",
        now: int = 0,
    ) -> PaymentSession:
        now = now or now_ms()
        gateway = self._gateways.get(provider)
        if gateway is None:
            raise DomainError(f"unknown provider {provider}", code=ErrorCode.NOT_FOUND)
        package = self._pricing.get_package(package_id)
        payment = Payment(
            payment_id=new_id("pay"),
            user_id=user_id,
            provider=provider,
            package_id=package.package_id,
            currency=package.currency,
            amount_minor=package.amount_minor,
            point_units=package.point_units,
            created_at=now,
        )
        session = gateway.create_session(
            PaymentSessionRequest(
                payment_id=payment.payment_id,
                user_id=user_id,
                package=package,
                success_url=success_url,
                cancel_url=cancel_url,
            )
        )
        payment.provider_order_id = session.provider_order_id
        with self._store.transaction() as tx:
            tx.insert(payment, payment.payment_id)
        return session

    def get_payment(self, payment_id: str, user_id: str) -> Payment:
        with self._store.transaction() as tx:
            payment = tx.get(Payment, payment_id)
            if payment is None:
                raise NotFound(f"payment {payment_id}")
            if payment.user_id != user_id:
                raise Forbidden("not your payment")
            return payment

    def record_webhook(self, provider: str, headers, raw_body: bytes, *, now: int = 0) -> str:
        """Verify + persist the event; returns the inbox key. Duplicates are absorbed."""
        now = now or now_ms()
        gateway = self._gateways.get(provider)
        if gateway is None:
            raise DomainError(f"unknown provider {provider}", code=ErrorCode.NOT_FOUND)
        event = gateway.verify_webhook(headers, raw_body)
        return self._inbox_event(
            provider=provider,
            event_id=event.event_id,
            event_type=event.event_type,
            payment_id=event.payment_id,
            payload={
                "amount_minor": event.amount_minor,
                "currency": event.currency,
                "provider_payment_id": event.provider_payment_id,
                "provider_capture_id": event.provider_capture_id,
                **event.payload,
            },
            now=now,
        )

    def capture_paypal(self, order_id: str, user_id: str, *, now: int = 0) -> str:
        """Server-side PayPal capture (§10.1); approval alone never posts points."""
        now = now or now_ms()
        gateway = self._gateways.get("paypal")
        if gateway is None:
            raise DomainError("paypal not configured", code=ErrorCode.NOT_FOUND)
        with self._store.transaction() as tx:
            matches = tx.query(Payment, where=("provider_order_id", "==", order_id))
        payment = next((p for p in matches if p.user_id == user_id), None)
        if payment is None:
            raise NotFound(f"order {order_id}")
        snapshot = gateway.capture(order_id, idempotency_key=f"capture:{payment.payment_id}")
        if snapshot.status != "completed":
            raise DomainError("capture not completed", code=ErrorCode.BACKEND_FAILED)
        return self._inbox_event(
            provider="paypal",
            event_id=f"capture:{snapshot.provider_capture_id}",
            event_type="capture.completed",
            payment_id=payment.payment_id,
            payload={
                "amount_minor": snapshot.amount_minor,
                "currency": snapshot.currency,
                "provider_payment_id": snapshot.provider_payment_id,
                "provider_capture_id": snapshot.provider_capture_id,
            },
            now=now,
        )

    def _inbox_event(self, *, provider: str, event_id: str, event_type: str, payment_id: str | None, payload: dict, now: int) -> str:
        key = f"{provider}:{event_id}"
        event = PaymentEvent(
            event_key=key,
            provider=provider,
            event_id=event_id,
            event_type=event_type,
            processing_status=PaymentEventStatus.RECEIVED,
            payment_id=payment_id,
            payload=payload,
            received_at=now,
            next_attempt_at=now,
        )
        with self._store.transaction() as tx:
            existing = tx.get(PaymentEvent, key)
            if existing is None:
                tx.insert(event, key)
        return key
