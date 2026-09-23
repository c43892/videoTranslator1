import httpx
import pytest
from fastapi.testclient import TestClient

from videotranslator.adapters.payments import StripePaymentGateway
from videotranslator.api.app import create_app
from videotranslator.domain.models import Payment, User, LedgerEntry
from .test_paid_accounts import stripe_event


def setup_checkout(container, user, monkeypatch, **override):
    session = container.billing.create_session(user.user_id, package_id='points_10_v1', provider='stripe')
    with container.store.transaction() as tx:
        p = tx.get(Payment, session.payment_id)
        p.provider_order_id = 'cs_test_saved'
        tx.put(p, p.payment_id)
    gateway = StripePaymentGateway('test', 'test-webhook')
    container.gateways['stripe'] = gateway
    data = dict(id='cs_test_saved', mode='payment', status='complete', payment_status='paid',
                amount_total=1000, currency='usd', client_reference_id=session.payment_id, payment_intent='pi_verified')
    data.update(override)
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, json=data)
    monkeypatch.setattr(httpx, 'get', get)
    client = TestClient(create_app(container), headers={'Authorization':'Bearer fake:u1'})
    return session.payment_id, calls, client


def test_missed_webhook_recovered_once_even_when_webhook_arrives_later(container, user, monkeypatch):
    pid, calls, client = setup_checkout(container, user, monkeypatch)
    url = f'/api/v1/billing/payments/{pid}/reconcile'
    assert client.post(url).json()['status'] == 'succeeded'
    assert client.post(url).json()['status'] == 'succeeded'
    assert calls == ['https://api.stripe.com/v1/checkout/sessions/cs_test_saved']
    headers, body = stripe_event(pid)
    container.billing.record_webhook('stripe', headers, body)
    container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User, user.user_id).point_balance_units == 1000
        assert len(tx.query(LedgerEntry)) == 1


@pytest.mark.parametrize('override', [dict(payment_status='unpaid'), dict(status='open'),
    dict(amount_total=999), dict(currency='eur'), dict(client_reference_id='another-payment')])
def test_unpaid_or_mismatched_checkout_never_credits(container, user, monkeypatch, override):
    pid, _, client = setup_checkout(container, user, monkeypatch, **override)
    client.post(f'/api/v1/billing/payments/{pid}/reconcile')
    with container.store.transaction() as tx:
        assert tx.get(User, user.user_id).point_balance_units == 0
        assert tx.get(Payment, pid).status == 'pending'
        assert tx.query(LedgerEntry) == []


def test_other_user_cannot_verify_or_choose_provider_session(container, user, monkeypatch):
    pid, calls, client = setup_checkout(container, user, monkeypatch)
    assert client.post(f'/api/v1/billing/payments/{pid}/reconcile',
        headers={'Authorization':'Bearer fake:other'}).status_code == 403
    assert calls == []
    assert client.post(f'/api/v1/billing/payments/{pid}/reconcile',
        json={'provider_order_id':'cs_attacker','amount_minor':99999}).json()['status'] == 'succeeded'
    assert calls == ['https://api.stripe.com/v1/checkout/sessions/cs_test_saved']
