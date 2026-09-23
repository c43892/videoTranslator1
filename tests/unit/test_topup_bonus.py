import pytest
from fastapi.testclient import TestClient
from videotranslator.api.app import create_app
from videotranslator.application.billing import StaticTopUpPricing
from videotranslator.config import Settings
from videotranslator.domain.models import User, Payment, LedgerEntry


@pytest.mark.parametrize('provider',['stripe','paypal'])
@pytest.mark.parametrize('package_id,paid,credited',[
    ('points_1_v1',100,100),('points_10_v2',1000,1100),
    ('points_50_v2',5000,6000),('points_100_v2',10000,13000),
])
def test_provider_charges_cash_and_credits_bonus_exactly_once(container,user,provider,package_id,paid,credited):
    container.billing._pricing=StaticTopUpPricing(Settings().topup_packages)
    session=container.billing.create_session(user.user_id,package_id=package_id,provider=provider)
    gateway=container.gateways[provider]
    request=gateway._sessions[session.payment_id]
    assert request.package.amount_minor==paid
    with container.store.transaction() as tx:
        payment=tx.get(Payment,session.payment_id)
        assert (payment.amount_minor,payment.point_units)==(paid,credited)
        assert tx.get(User,user.user_id).point_balance_units==0
    for _ in range(2):
        if provider=='paypal':
            container.billing.capture_paypal(session.provider_order_id,user.user_id)
        else:
            headers,body=gateway.make_webhook(session.payment_id)
            container.billing.record_webhook(provider,headers,body)
        container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units==credited
        entries=tx.query(LedgerEntry)
        assert len(entries)==1 and entries[0].delta_units==credited


def test_pending_old_payment_keeps_original_credit_after_catalog_upgrade(container,user):
    session=container.billing.create_session(user.user_id,package_id='points_10_v1',provider='stripe')
    container.billing._pricing=StaticTopUpPricing(Settings().topup_packages)
    headers,body=container.gateways['stripe'].make_webhook(session.payment_id)
    container.billing.record_webhook('stripe',headers,body)
    container.inbox.process_due()
    with container.store.transaction() as tx:
        assert tx.get(User,user.user_id).point_balance_units==1000


def test_catalog_exposes_cash_and_wallet_credit(container,user):
    container.billing._pricing=StaticTopUpPricing(Settings().topup_packages)
    client=TestClient(create_app(container),headers={'Authorization':'Bearer fake:u1'})
    result=client.get('/api/v1/billing/packages')
    assert result.status_code==200
    assert [(p['amount_minor'],p['point_units']) for p in result.json()['packages']]==[
        (100,100),(1000,1100),(5000,6000),(10000,13000)]
