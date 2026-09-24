"""Durable active pricing and immutable quote snapshots."""

from ..domain.models import PricingConfig, PricingState, now_ms
from ..ids import new_id
from ..domain.enums import StaleVersion


def public_price(price):
    return {"pricing_version": price.pricing_version, "currency": "USD",
            "rate_cents_per_minute": price.point_units_per_minute,
            "minimum_cents": price.minimum_point_units,
            "billing_increment_cents": price.billing_increment_units}


class PricingService:
    def __init__(self, store, fallback):
        self.store, self.fallback = store, fallback

    def resolve(self, tx, version=""):
        if version:
            price = tx.get(PricingConfig, version)
            if price is None:
                raise StaleVersion("price_changed")
            return price
        state = tx.get(PricingState, "active")
        if state:
            return self.resolve(tx, state.pricing_version)
        return self.fallback()

    def current(self, version=""):
        with self.store.transaction() as tx:
            return self.resolve(tx, version)

    def snapshot(self, tx):
        price = self.resolve(tx)
        if tx.get(PricingConfig, price.pricing_version) is None:
            tx.insert(price, price.pricing_version)
        return price

    def update(self, *, expected_version, rate, minimum, increment, actor):
        if not actor or any(type(n) is not int or not 1 <= n <= 10000 for n in (rate, minimum, increment)):
            raise ValueError("invalid_pricing")
        with self.store.transaction() as tx:
            if self.resolve(tx).pricing_version != expected_version:
                raise StaleVersion("price_changed")
            price = PricingConfig(pricing_version=new_id("price"), point_units_per_minute=rate,
                minimum_point_units=minimum, billing_increment_units=increment,
                rounding="ceil_final_billing_increment", active_from=now_ms(), updated_by=actor)
            tx.insert(price, price.pricing_version)
            state = tx.get(PricingState, "active")
            if state:
                state.pricing_version = price.pricing_version
                tx.put(state, "active")
            else:
                tx.insert(PricingState(pricing_version=price.pricing_version), "active")
        return price
