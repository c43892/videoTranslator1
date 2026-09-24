from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from videotranslator.api.app import create_app
from videotranslator.bootstrap import build_container
from videotranslator.domain.models import PricingConfig
from videotranslator.domain.pricing import quote_job


def test_admin_authorization_validation_and_stale_writes(container):
    container.settings = replace(container.settings, admin_user_ids=("owner",))
    http = TestClient(create_app(container))
    path = "/api/v1/admin/pricing"
    owner = {"Authorization": "Bearer fake:owner"}
    assert http.get(path).status_code == 401
    assert http.get(path, headers={"Authorization": "Bearer fake:other"}).status_code == 403
    body = dict(expected_version="job-v1", rate_cents_per_minute=20, minimum_cents=10, billing_increment_cents=10)
    assert http.put(path, json=body, headers={"Authorization": "Bearer fake:other"}).status_code == 403
    assert http.put(path, json={**body, "minimum_cents": 0}, headers=owner).status_code == 422
    assert http.put(path, json={**body, "rate_cents_per_minute": 0.2}, headers=owner).status_code == 422
    saved = http.put(path, json=body, headers=owner)
    assert saved.status_code == 200
    assert http.put(path, json=body, headers=owner).status_code == 409
    config = http.get("/api/v1/chat/config").json()
    assert config["rate_cents_per_minute"] == 20
    assert config["pricing_version"] == saved.json()["pricing_version"]
    assert http.get("/api/v1/me", headers=owner).json()["is_admin"]
    with container.store.transaction() as tx:
        assert tx.get(PricingConfig, config["pricing_version"]).updated_by == "owner"
        assert tx.get(PricingConfig, "job-v1").point_units_per_minute == 100


def test_price_persists_after_restart_and_seed(container, tmp_path):
    settings = replace(container.settings, store_path=str(tmp_path / "prices.db"))
    first = build_container(settings)
    price = first.funding.pricing.update(expected_version="job-v1", rate=20, minimum=10, increment=10, actor="owner")
    second = build_container(settings)
    second.seed()
    assert second.funding.pricing.current() == price


@pytest.mark.parametrize("duration,amount", [(1,10),(30000,10),(30001,20),(60000,20),(61000,30),(90000,30)])
def test_new_price_rounds_total_only(container, duration, amount):
    price = container.funding.pricing.update(expected_version="job-v1", rate=20, minimum=10, increment=10, actor="owner")
    assert quote_job(price, duration, "video").point_units == amount
