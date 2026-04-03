"""Tests for the pure dynamics functions (weather, interest, markets, yield, spoilage)."""

import pytest
import numpy as np

from cropRL.config import EnvConfig
from cropRL.dynamics import (
    generate_rainfall,
    calculate_interest_rate,
    generate_market_prices,
    calculate_yield,
    calculate_expected_yield_potential,
    apply_spoilage,
)


class TestRainfall:
    def test_monsoon_high(self, config):
        rainfalls = [
            generate_rainfall(7, config, np.random.default_rng(i))
            for i in range(100)
        ]
        assert np.mean(rainfalls) > 0.6

    def test_summer_low(self, config):
        rainfalls = [
            generate_rainfall(5, config, np.random.default_rng(i))
            for i in range(100)
        ]
        assert np.mean(rainfalls) < 0.3

    def test_bounded_all_months(self, config, rng):
        for month in range(1, 13):
            rain = generate_rainfall(month, config, rng)
            assert 0.0 <= rain <= 1.0


class TestInterestRate:
    def test_planting_season_premium(self):
        rate = calculate_interest_rate(0.08, 6, 0.8, 0.6)
        assert rate == 0.08 + 0.03

    def test_harvest_season_discount(self):
        rate = calculate_interest_rate(0.08, 10, 0.3, 0.3)
        assert rate == 0.08 - 0.02

    def test_drought_risk_premium(self):
        rate = calculate_interest_rate(0.08, 3, 0.1, 0.6)
        assert rate == 0.08 + 0.05


class TestMarketPrices:
    def test_reproducible(self, config):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert generate_market_prices(6, config, rng1) == generate_market_prices(6, config, rng2)

    def test_positive_all_months(self, config, rng):
        for month in range(1, 13):
            prices = generate_market_prices(month, config, rng)
            assert all(p > 0 for p in prices)

    def test_pre_monsoon_premium(self, config):
        """Apr-May should have 1.15x seasonal multiplier (higher prices)."""
        prices_apr = []
        prices_jan = []
        for i in range(200):
            r = np.random.default_rng(i)
            prices_apr.append(generate_market_prices(4, config, r)[0])
            prices_jan.append(generate_market_prices(1, config, r)[0])
        # April (pre-monsoon) should be higher on average than January
        assert np.mean(prices_apr) > np.mean(prices_jan)


class TestYield:
    def test_fallow_zero(self, config):
        assert calculate_yield(0, 0, 0.5, 0.5, False, config) == 0.0

    def test_mature_good_conditions(self, config):
        y = calculate_yield(1, 4, 0.6, 0.6, False, config)
        assert y == pytest.approx(8.0 * 1.2 * 1.0 * 1.0)

    def test_early_harvest_penalty(self, config):
        y = calculate_yield(1, 1, 0.5, 0.6, False, config)
        assert y < 8.0 * 0.5

    def test_irrigation_helps_drought(self, config):
        y_dry = calculate_yield(1, 4, 0.5, 0.0, False, config)
        y_irr = calculate_yield(1, 4, 0.5, 0.0, True, config)
        assert y_irr > y_dry


class TestExpectedYieldPotential:
    def test_fallow_zero(self, config):
        assert calculate_expected_yield_potential(0, 0, 0.5, 0.5, config) == 0.0

    def test_in_unit_range(self, config):
        p = calculate_expected_yield_potential(1, 4, 0.8, 0.6, config)
        assert 0.0 <= p <= 1.0


class TestSpoilage:
    def test_fresh_kept(self):
        amount, spoiled = apply_spoilage(3, 5.0, 6)
        assert amount == 5.0
        assert spoiled is False

    def test_rotten_zeroed(self):
        amount, spoiled = apply_spoilage(7, 5.0, 6)
        assert amount == 0.0
        assert spoiled is True

    def test_boundary_not_spoiled(self):
        amount, spoiled = apply_spoilage(6, 5.0, 6)
        assert amount == 5.0
        assert spoiled is False
