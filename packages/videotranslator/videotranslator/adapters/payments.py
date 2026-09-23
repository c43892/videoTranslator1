"""Stripe and PayPal gateways over plain HTTP (§6.4).

httpx is imported lazily so the test profile never needs the dependency or
the network. Stripe confirms via verified webhooks; PayPal Orders v2
confirms via server-side capture (§10.1).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from decimal import Decimal
from typing import Mapping

from ..domain.enums import BackendError, FailureClass
from ..ports import PaymentEventData, PaymentSession, PaymentSessionRequest, PaymentSnapshot


def _httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise BackendError("httpx is required for live payment gateways") from exc
    return httpx


class StripePaymentGateway:
    provider = "stripe"

    def __init__(self, secret_key: str, webhook_secret: str, *, base_url: str = "https://api.stripe.com"):
        self._key = secret_key
        self._webhook_secret = webhook_secret
        self._base = base_url

    def create_session(self, request: PaymentSessionRequest) -> PaymentSession:
        httpx = _httpx()
        resp = httpx.post(
            f"{self._base}/v1/checkout/sessions",
            auth=(self._key, ""),
            headers={"Idempotency-Key": request.payment_id},
            data={
                "mode": "payment",
                "success_url": request.success_url or "https://localhost/return",
                "cancel_url": request.cancel_url or "https://localhost/return",
                "client_reference_id": request.payment_id,
                "line_items[0][quantity]": "1",
                "line_items[0][price_data][currency]": request.package.currency.lower(),
                "line_items[0][price_data][unit_amount]": str(request.package.amount_minor),
                "line_items[0][price_data][product_data][name]": f"USD {request.package.point_units / 100:.2f} translation balance (non-refundable)",
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            raise BackendError(f"stripe checkout failed: {resp.status_code}")
        data = resp.json()
        return PaymentSession(
            payment_id=request.payment_id,
            provider=self.provider,
            confirmation_mode="webhook",
            redirect_url=data["url"],
            provider_order_id=data["id"],
        )

    def verify_webhook(self, headers: Mapping[str, str], raw_body: bytes) -> PaymentEventData:
        header = headers.get("stripe-signature", "")
        parts = [p.strip().split("=", 1) for p in header.split(",") if "=" in p]
        timestamp = next((v for k, v in parts if k == "t"), "")
        signatures = [v for k, v in parts if k == "v1"]
        expected = hmac.new(
            self._webhook_secret.encode(), f"{timestamp}.".encode() + raw_body, hashlib.sha256
        ).hexdigest()
        if not timestamp.isdigit() or not any(hmac.compare_digest(expected, sig) for sig in signatures):
            raise BackendError("bad stripe signature", failure_class=FailureClass.PERMANENT)
        if abs(time.time() - int(timestamp)) > 300:
            raise BackendError("stale stripe webhook", failure_class=FailureClass.PERMANENT)
        event = json.loads(raw_body)
        obj = event.get("data", {}).get("object", {})
        return PaymentEventData(
            provider=self.provider,
            event_id=event["id"],
            event_type=event.get("type", "") if obj.get("payment_status") == "paid" else "ignored",
            payment_id=obj.get("client_reference_id"),
            provider_payment_id=obj.get("payment_intent") or obj.get("id"),
            provider_capture_id=None,
            amount_minor=int(obj.get("amount_total", 0)),
            currency=str(obj.get("currency", "usd")).upper(),
            payload={},
        )

    def get_payment(self, provider_payment_id: str) -> PaymentSnapshot:
        httpx = _httpx()
        if provider_payment_id.startswith('cs_'):
            resp = httpx.get(f'{self._base}/v1/checkout/sessions/{provider_payment_id}',
                             auth=(self._key, ''), timeout=15)
            if resp.status_code >= 400:
                raise BackendError(f'stripe checkout lookup failed: {resp.status_code}')
            data = resp.json()
            if data.get('id') != provider_payment_id or data.get('mode') != 'payment':
                raise BackendError('stripe checkout mismatch', failure_class=FailureClass.PERMANENT)
            return PaymentSnapshot(
                provider_payment_id=data.get('payment_intent') or provider_payment_id,
                status='completed' if data.get('status') == 'complete' and data.get('payment_status') == 'paid' else 'pending',
                amount_minor=int(data.get('amount_total', 0)),
                currency=str(data.get('currency', '')).upper(),
                payment_reference=data.get('client_reference_id'),
            )
        resp = httpx.get(f"{self._base}/v1/payment_intents/{provider_payment_id}", auth=(self._key, ""), timeout=30)
        if resp.status_code >= 400:
            raise BackendError(f"stripe lookup failed: {resp.status_code}")
        data = resp.json()
        return PaymentSnapshot(
            provider_payment_id=provider_payment_id,
            status="completed" if data.get("status") == "succeeded" else "pending",
            amount_minor=int(data.get("amount", 0)),
            currency=str(data.get("currency", "usd")).upper(),
        )

    def refund(self, provider_payment_id: str, amount_minor: int) -> bool:
        raise BackendError("top-ups are non-refundable", failure_class=FailureClass.PERMANENT)


class PayPalPaymentGateway:
    provider = "paypal"

    def __init__(self, client_id: str, client_secret: str, *, base_url: str = "https://api-m.sandbox.paypal.com", webhook_id: str = ""):
        self._client_id = client_id
        self._client_secret = client_secret
        self._base = base_url
        self._webhook_id = webhook_id

    def _token(self) -> str:
        httpx = _httpx()
        resp = httpx.post(
            f"{self._base}/v1/oauth2/token",
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise BackendError("paypal auth failed")
        return resp.json()["access_token"]

    def create_session(self, request: PaymentSessionRequest) -> PaymentSession:
        httpx = _httpx()
        resp = httpx.post(
            f"{self._base}/v2/checkout/orders",
            headers={"Authorization": f"Bearer {self._token()}", "PayPal-Request-Id": request.payment_id},
            json={
                "intent": "CAPTURE",
                "purchase_units": [
                    {
                        "reference_id": request.payment_id,
                        "custom_id": request.payment_id,
                        "description": "Translation balance (non-refundable)",
                        "amount": {
                            "currency_code": request.package.currency,
                            "value": f"{request.package.amount_minor / 100:.2f}",
                        },
                    }
                ],
                "payment_source": {
                    "paypal": {
                        "experience_context": {
                            "return_url": request.success_url or "https://localhost/return",
                            "cancel_url": request.cancel_url or "https://localhost/return",
                        }
                    }
                },
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            raise BackendError(f"paypal order failed: {resp.status_code}")
        data = resp.json()
        approval = next(l["href"] for l in data["links"] if l["rel"] in {"payer-action", "approve"})
        return PaymentSession(
            payment_id=request.payment_id,
            provider=self.provider,
            confirmation_mode="server_capture",
            redirect_url=approval,
            provider_order_id=data["id"],
        )

    def capture(self, provider_order_id: str, idempotency_key: str) -> PaymentSnapshot:
        httpx = _httpx()
        resp = httpx.post(
            f"{self._base}/v2/checkout/orders/{provider_order_id}/capture",
            headers={"Authorization": f"Bearer {self._token()}", "PayPal-Request-Id": idempotency_key},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise BackendError(f"paypal capture failed: {resp.status_code}")
        data = resp.json()
        unit = data["purchase_units"][0]
        capture = unit["payments"]["captures"][0]
        amount = capture["amount"]
        return PaymentSnapshot(
            provider_payment_id=data["id"],
            status="completed" if capture.get("status") == "COMPLETED" else "pending",
            amount_minor=int(Decimal(amount["value"]) * 100),
            currency=amount["currency_code"],
            user_id=unit.get("reference_id"),
            provider_capture_id=capture["id"],
        )

    def verify_webhook(self, headers: Mapping[str, str], raw_body: bytes) -> PaymentEventData:
        event = json.loads(raw_body)
        if not self._webhook_id:
            raise BackendError("PayPal webhook ID not configured")
        response = _httpx().post(f"{self._base}/v1/notifications/verify-webhook-signature",
            headers={"Authorization": f"Bearer {self._token()}"}, timeout=30,
            json={"auth_algo": headers.get("paypal-auth-algo"), "cert_url": headers.get("paypal-cert-url"),
                "transmission_id": headers.get("paypal-transmission-id"), "transmission_sig": headers.get("paypal-transmission-sig"),
                "transmission_time": headers.get("paypal-transmission-time"), "webhook_id": self._webhook_id,
                "webhook_event": event})
        if response.status_code >= 400 or response.json().get("verification_status") != "SUCCESS":
            raise BackendError("bad PayPal signature", failure_class=FailureClass.PERMANENT)
        resource = event.get("resource", {})
        amount = resource.get("amount", {})
        return PaymentEventData(
            provider=self.provider,
            event_id=event["id"],
            event_type=event.get("event_type", "") if resource.get("status") == "COMPLETED" else "ignored",
            payment_id=resource.get("custom_id") or resource.get("reference_id"),
            provider_payment_id=resource.get("id"),
            provider_capture_id=resource.get("id") if "CAPTURE" in event.get("event_type", "") else None,
            amount_minor=int(Decimal(amount.get("value", "0")) * 100),
            currency=amount.get("currency_code", "USD"),
            payload={},
        )

    def get_payment(self, provider_payment_id: str) -> PaymentSnapshot:
        httpx = _httpx()
        resp = httpx.get(
            f"{self._base}/v2/checkout/orders/{provider_payment_id}",
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=30,
        )
        if resp.status_code >= 400:
            raise BackendError("paypal lookup failed")
        data = resp.json()
        unit = data["purchase_units"][0]
        return PaymentSnapshot(
            provider_payment_id=provider_payment_id,
            status="completed" if data.get("status") == "COMPLETED" else "pending",
            amount_minor=int(Decimal(unit["amount"]["value"]) * 100),
            currency=unit["amount"]["currency_code"],
        )

    def refund(self, provider_payment_id: str, amount_minor: int) -> bool:
        raise BackendError("top-ups are non-refundable", failure_class=FailureClass.PERMANENT)
