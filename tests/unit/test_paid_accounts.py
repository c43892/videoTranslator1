"""Money and identity boundaries: exercise real gateway verification with mocked HTTP."""
import hashlib
import hmac
import json
import time
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from videotranslator.adapters.payments import StripePaymentGateway, PayPalPaymentGateway
from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.config import Settings
from videotranslator.domain.enums import BackendError, DomainError
from videotranslator.domain.models import Payment, PaymentEvent, User, LedgerEntry
from videotranslator.domain.pricing import quote_job


@pytest.mark.parametrize("duration,cents", [(1,10),(30000,10),(30001,20),(60000,20),(60001,30),(90000,30),(90001,40),(600000,200),(1800000,600)])
def test_default_price_is_twenty_cents_per_minute_for_audio_and_video(duration, cents):
    for media in ("audio", "video"):
        assert quote_job(Settings().pricing, duration, media).point_units == cents


def stripe_event(payment_id, *, paid=True, amount=1000, timestamp=None, event_id="evt_1"):
    body = json.dumps({"id":event_id,"type":"checkout.session.completed", "data":{"object":{
        "client_reference_id":payment_id,"payment_status":"paid" if paid else "unpaid",
        "amount_total":amount,"currency":"usd", "payment_intent":"pi_1"}}}).encode()
    timestamp = int(time.time()) if timestamp is None else timestamp
    signature = hmac.new(b"test-webhook", f"{timestamp}.".encode()+body, hashlib.sha256).hexdigest()
    return {"stripe-signature":f"t={timestamp},v1=old-secret-signature,v1={signature}"},body


def test_verified_payment_once_and_unpaid_never_credits(container, user):
    session = container.billing.create_session(user.user_id, package_id="points_10_v1", provider="stripe")
    container.gateways["stripe"] = StripePaymentGateway("test", "test-webhook")
    headers, body = stripe_event(session.payment_id, paid=False)
    assert container.billing.record_webhook("stripe", headers, body) == "ignored"
    with container.store.transaction() as tx:
        assert tx.query(PaymentEvent) == [] and tx.get(User,user.user_id).point_balance_units == 0
    for eid in ("evt_1", "evt_1", "evt_2"):
        headers, body = stripe_event(session.payment_id, event_id=eid)
        container.billing.record_webhook("stripe", headers, body)
        container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 1000
        assert len(tx.query(LedgerEntry)) == 1


def test_stripe_rejects_bad_signature_stale_delivery_and_wrong_amount(container, user):
    gateway = StripePaymentGateway("test", "test-webhook")
    headers, body = stripe_event("pay_1", timestamp=1)
    with pytest.raises(BackendError): gateway.verify_webhook(headers, body)
    headers, body = stripe_event("pay_1")
    with pytest.raises(BackendError): gateway.verify_webhook(headers, body+b" ")
    session = container.billing.create_session(user.user_id,package_id="points_10_v1",provider="stripe")
    container.gateways["stripe"] = gateway
    headers, body = stripe_event(session.payment_id,amount=1)
    container.billing.record_webhook("stripe",headers,body)
    container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 0
        assert tx.query(LedgerEntry) == []


def test_checkout_idempotency_and_server_owned_return_url(container, user):
    http = TestClient(create_app(container), headers={"Authorization":"Bearer fake:u1"})
    body={"package_id":"points_10_v1","provider":"stripe","idempotency_key":"click-1", "success_url":"https://evil.test"}
    a=http.post("/api/v1/billing/sessions",json=body)
    b=http.post("/api/v1/billing/sessions",json=body)
    assert a.status_code == b.status_code == 201 and a.json() == b.json()
    request=container.gateways["stripe"]._sessions[a.json()["payment_id"]]
    assert request.success_url.startswith(container.settings.public_app_url+"/?checkout=return&payment_id=")
    assert "evil.test" not in request.success_url
    with container.store.transaction() as tx:
        assert len(tx.query(Payment)) == 1 and tx.get(User,user.user_id).point_balance_units == 0
    assert http.post("/api/v1/billing/sessions",json={**body,"package_id":"points_50_v1"}).status_code == 400


def test_provider_mismatch_cannot_credit(container, user):
    session=container.billing.create_session(user.user_id,package_id="points_10_v1",provider="stripe")
    container.gateways["paypal"]._sessions[session.payment_id] = container.gateways["stripe"]._sessions[session.payment_id]
    headers,body=container.gateways["paypal"].make_webhook(session.payment_id)
    container.billing.record_webhook("paypal",headers,body)
    container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units == 0


def test_paypal_signature_is_verified_remotely_and_fails_closed(monkeypatch):
    seen=[]
    def post(url, **kwargs):
        if url.endswith("/token"): return httpx.Response(200,json={"access_token":"access"})
        seen.append(kwargs["json"])
        return httpx.Response(200,json={"verification_status":"FAILURE"})
    monkeypatch.setattr(httpx,"post",post)
    gateway=PayPalPaymentGateway("client","secret",webhook_id="WH-1")
    body=json.dumps({"id":"evt-1","event_type":"PAYMENT.CAPTURE.COMPLETED","resource":{"status":"COMPLETED","custom_id":"pay-1","amount":{"value":"10.00","currency_code":"USD"}}}).encode()
    with pytest.raises(BackendError): gateway.verify_webhook({},body)
    assert seen[0]["webhook_id"] == "WH-1"
    def verified(url, **kwargs):
        return httpx.Response(200,json={"access_token":"access"} if url.endswith("/token") else {"verification_status":"SUCCESS"})
    monkeypatch.setattr(httpx,"post",verified)
    event=gateway.verify_webhook({},body)
    assert event.payment_id == "pay-1" and event.amount_minor == 1000


def test_refunds_cannot_send_money():
    for gateway in (StripePaymentGateway("key","hook"),PayPalPaymentGateway("id","secret")):
        with pytest.raises(BackendError,match="non-refundable"): gateway.refund("payment",1000)


def test_demo_identities_cannot_access_live_checkout(tmp_path):
    container=build_container(Settings(profile="local-ui",auth_mode="demo",payment_mode="live",local_storage_dir=str(tmp_path)))
    assert container.gateways == {}


def test_auth_and_ownership_enforced_on_money(container, user):
    http=TestClient(create_app(container))
    assert http.get("/api/v1/me").status_code == 401
    body={"package_id":"points_10_v1","provider":"stripe"}
    assert http.post("/api/v1/billing/sessions",json=body,headers={"Authorization":"Bearer fake:u1:unverified"}).status_code == 403
    session=container.billing.create_session(user.user_id,package_id="points_10_v1",provider="stripe")
    assert http.get(f"/api/v1/billing/payments/{session.payment_id}",headers={"Authorization":"Bearer fake:other"}).status_code == 403


def test_firebase_is_default_and_emulator_is_not_used(tmp_path):
    from videotranslator.adapters.auth import LazyFirebaseIdentityVerifier
    container=build_container(Settings(profile="local-ui",local_storage_dir=str(tmp_path)))
    assert isinstance(container.identity,LazyFirebaseIdentityVerifier) and not container.gateways
    assert TestClient(create_app(container)).get("/api/v1/chat/config").json()["auth_mode"] == "firebase"


def test_live_billing_requires_durable_storage():
    with pytest.raises(ValueError, match="persistent"):
        build_container(Settings(profile="local-ui", payment_mode="live"))


def test_checkout_timeout_retry_reuses_saved_payment(container, user):
    http=TestClient(create_app(container),headers={"Authorization":"Bearer fake:u1"})
    gateway=container.gateways["stripe"]
    original=gateway.create_session
    gateway.create_session=lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("offline"))
    body={"package_id":"points_10_v1","provider":"stripe","idempotency_key":"retry-timeout"}
    assert http.post("/api/v1/billing/sessions",json=body).status_code == 503
    gateway.create_session=original
    assert http.post("/api/v1/billing/sessions",json=body).status_code == 201
    with container.store.transaction() as tx:
        assert len(tx.query(Payment)) == 1 and tx.get(User,"u1").point_balance_units == 0
