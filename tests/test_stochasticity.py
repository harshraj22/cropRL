"""Tests for stochastic features: yield noise, price autocorrelation, demand shocks."""

import numpy as np
import pytest

from cropRL.config import EnvConfig
from cropRL.dynamics import calculate_yield, generate_market_prices


# ── Yield Noise ────────────────────────────────────────────────


class TestYieldNoise:
    """Tests for Gaussian noise on harvest yield."""

    def test_deterministic_without_rng(self):
        """calculate_yield with rng=None should be fully deterministic."""
        cfg = EnvConfig()
        # New signature: crop_type, age, nitrogen, water_level, month, config, rng
        y1 = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=None)
        y2 = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=None)
        assert y1 == y2

    def test_noisy_with_rng(self):
        """With rng, repeated calls should vary."""
        cfg = EnvConfig(yield_sigma=0.15)
        rng = np.random.default_rng(42)
        yields = [
            calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=rng)
            for _ in range(50)
        ]
        assert len(set(yields)) > 1, "Yields should vary with rng"

    def test_yield_never_negative(self):
        """Even with high noise, yield should be clamped >= 0."""
        cfg = EnvConfig(yield_sigma=0.50)
        rng = np.random.default_rng(99)
        for _ in range(200):
            y = calculate_yield(3, 3, 0.3, 0.3, 10, cfg, rng=rng)
            assert y >= 0.0

    def test_noise_is_bounded(self):
        """Noise is clamped to ±3σ, so yield should stay within bounds."""
        cfg = EnvConfig(yield_sigma=0.10)
        rng = np.random.default_rng(0)
        base_det = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=None)
        for _ in range(200):
            y = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=rng)
            assert y >= base_det * 0.69  # small float tolerance
            assert y <= base_det * 1.31

    def test_zero_sigma_means_deterministic(self):
        """yield_sigma=0 should produce identical results with or without rng."""
        cfg = EnvConfig(yield_sigma=0.0)
        rng = np.random.default_rng(42)
        y_no_rng = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=None)
        y_with_rng = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=rng)
        assert y_no_rng == y_with_rng


# ── Price Autocorrelation ──────────────────────────────────────


class TestPriceAutocorrelation:
    """Tests for mean-reverting random walk market prices."""

    def test_autocorrelation_uses_prev_prices(self):
        """With autocorrelation enabled, prices should correlate with previous."""
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            price_reversion_speed=0.3,
            market_price_sigma=0.05,
            demand_shock_probability=0.0,
        )
        rng = np.random.default_rng(42)
        high_prev = (2000.0, 1500.0, 900.0)
        prices_from_high = generate_market_prices(6, cfg, rng, prev_prices=high_prev)

        rng2 = np.random.default_rng(42)
        low_prev = (500.0, 300.0, 150.0)
        prices_from_low = generate_market_prices(6, cfg, rng2, prev_prices=low_prev)

        assert prices_from_high[0] > prices_from_low[0]

    def test_independent_without_prev_prices(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_market_prices(6, cfg, rng1, prev_prices=None)
        p2 = generate_market_prices(6, cfg, rng2, prev_prices=None)
        assert p1 == p2

    def test_disabled_autocorrelation(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_market_prices(6, cfg, rng1, prev_prices=(2000, 2000, 2000))
        p2 = generate_market_prices(6, cfg, rng2, prev_prices=(100, 100, 100))
        assert p1 == p2

    def test_prices_stay_clamped(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            price_max_multiplier=2.5,
            demand_shock_probability=0.0,
        )
        rng = np.random.default_rng(42)
        extreme_prev = (5000.0, 5000.0, 5000.0)
        prices = generate_market_prices(6, cfg, rng, prev_prices=extreme_prev)
        for i, p in enumerate(prices):
            base = cfg.base_market_prices[i + 1]
            assert p <= base * cfg.price_max_multiplier + 0.01
            assert p >= base * cfg.price_min_multiplier - 0.01


# ── Demand Shocks ──────────────────────────────────────────────


class TestDemandShocks:
    """Tests for rare demand shock events."""

    def test_shocks_occur_over_many_draws(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.5,
            demand_shock_magnitude=(0.3, 0.6),
        )
        rng = np.random.default_rng(42)
        base_prices = []
        for _ in range(100):
            p = generate_market_prices(6, cfg, rng)
            base_prices.append(p)

        corn_prices = [p[0] for p in base_prices]
        base_corn = cfg.base_market_prices[1]
        large_deviations = [
            abs(p - base_corn) / base_corn for p in corn_prices
        ]
        assert max(large_deviations) > 0.15

    def test_no_shocks_when_disabled(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_market_prices(6, cfg, rng1)
        p2 = generate_market_prices(6, cfg, rng2)
        assert p1 == p2

    def test_shocked_prices_stay_clamped(self):
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=1.0,
            demand_shock_magnitude=(0.5, 0.6),
            price_max_multiplier=2.5,
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            prices = generate_market_prices(6, cfg, rng)
            for i, p in enumerate(prices):
                base = cfg.base_market_prices[i + 1]
                assert p <= base * cfg.price_max_multiplier + 0.01
                assert p >= base * cfg.price_min_multiplier - 0.01
