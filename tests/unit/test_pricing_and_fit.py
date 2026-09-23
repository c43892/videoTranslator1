"""§18.1: Decimal duration math, ceil_final_point_unit, atempo direction, fit policy."""

from __future__ import annotations

import pytest

from videotranslator.domain.duration_fit import (
    DurationPolicy,
    FitAction,
    atempo_factor,
    borrowable_gap_ms,
    decide_fit,
)
from videotranslator.domain.enums import DomainError
from videotranslator.domain.models import PricingConfig
from videotranslator.domain.pricing import (
    MEDIA_HARD_LIMIT_MS,
    max_runtime_seconds,
    probe_duration_to_ms,
    quote_job,
)

CONFIG = PricingConfig(pricing_version="job-v1", point_units_per_minute=100, minimum_point_units=1)


class TestProbeDuration:
    def test_decimal_ceiling_to_ms(self):
        assert probe_duration_to_ms("60.1") == 60_100
        assert probe_duration_to_ms("615.199743") == 615_200
        assert probe_duration_to_ms("0.0001") == 1  # conservative ceiling

    def test_no_binary_float_roundtrip(self):
        # 615.199743 as a float would be 615.199742999...; Decimal keeps 615200 ms
        assert probe_duration_to_ms("615.199743") == 615_200

    def test_rejects_garbage(self):
        with pytest.raises(DomainError):
            probe_duration_to_ms("not-a-number")
        with pytest.raises(DomainError):
            probe_duration_to_ms("-3")


class TestQuote:
    def test_examples_from_spec(self):
        assert quote_job(CONFIG, 30_000, "video").point_units == 50
        assert quote_job(CONFIG, 60_000, "video").point_units == 100
        assert quote_job(CONFIG, 90_000, "video").point_units == 150
        assert quote_job(CONFIG, 600_000, "video").point_units == 1000

    def test_single_ceiling_at_final_unit(self):
        # 60.1s → 60100ms → ceil(60100×100/60000) = 101 units
        assert quote_job(CONFIG, 60_100, "video").point_units == 101

    def test_minimum_units(self):
        assert quote_job(CONFIG, 1, "audio").point_units == 1

    def test_hard_limit_rejected(self):
        with pytest.raises(DomainError):
            quote_job(CONFIG, MEDIA_HARD_LIMIT_MS + 1, "video")

    def test_runtime_cap_formula(self):
        # ceil(615.2 × 4 + 900) = ceil(3360.8) = 3361
        assert max_runtime_seconds(615_200) == 3_361
        assert max_runtime_seconds(1_000) == 1_800
        assert max_runtime_seconds(MEDIA_HARD_LIMIT_MS) <= 14_400


class TestDurationFit:
    POLICY = DurationPolicy()

    def test_atempo_direction(self):
        # 8s squeezed into 5s → 1.6; the legacy bug produced 0.625
        assert atempo_factor(8000, 5000) == pytest.approx(1.6)
        assert atempo_factor(5000, 5000) == pytest.approx(1.0)

    def test_borrow_bounds(self):
        assert borrowable_gap_ms(10_000, 2_000, self.POLICY) == 500  # silence cap
        assert borrowable_gap_ms(1_000, 10_000, self.POLICY) == 200  # 20% ratio cap
        assert borrowable_gap_ms(10_000, -50, self.POLICY) == 0  # overlap → no borrow

    def test_pad_when_shorter(self):
        d = decide_fit(2_000, 3_000, 0, 0, self.POLICY)
        assert d.action == FitAction.PAD

    def test_fit_stays_inside_original_window_even_with_gap(self):
        d = decide_fit(3_300, 3_000, 500, 0, self.POLICY)
        assert d.action == FitAction.SPEED_UP
        assert d.atempo == pytest.approx(1.1)
        assert d.available_duration_ms == 3000
        assert d.borrowed_gap_ms == 0

    def test_speed_up_within_soft_limit(self):
        d = decide_fit(3_500, 3_000, 0, 0, self.POLICY)
        assert d.action == FitAction.SPEED_UP
        assert d.atempo == pytest.approx(3_500 / 3_000)

    def test_exceeding_old_limits_still_compresses_without_retranslation(self):
        over_soft = decide_fit(4_000, 3_000, 0, 0, self.POLICY)
        assert over_soft.action == FitAction.SPEED_UP
        assert over_soft.atempo == pytest.approx(4/3)
        still_over = decide_fit(4_200, 3_000, 0, 2, self.POLICY)
        assert still_over.action == FitAction.SPEED_UP
        assert still_over.atempo == pytest.approx(1.4)

    def test_hard_limit_speed_up_after_compactions(self):
        d = decide_fit(3_900, 3_000, 0, 2, self.POLICY)  # ratio 1.30 ≤ 1.35
        assert d.action == FitAction.SPEED_UP
