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
        y1 = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=None)
        y2 = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=None)
        assert y1 == y2

    def test_noisy_with_rng(self):
        """With rng, repeated calls should vary."""
        cfg = EnvConfig(yield_sigma=0.15)
        rng = np.random.default_rng(42)
        yields = [
            calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=rng)
            for _ in range(50)
        ]
        assert len(set(yields)) > 1, "Yields should vary with rng"

    def test_yield_never_negative(self):
        """Even with high noise, yield should be clamped >= 0."""
        cfg = EnvConfig(yield_sigma=0.50)  # very high noise
        rng = np.random.default_rng(99)
        for _ in range(200):
            y = calculate_yield(3, 3, 0.3, 0.1, False, cfg, rng=rng)
            assert y >= 0.0

    def test_noise_is_bounded(self):
        """Noise is clamped to ±3σ, so yield should stay within bounds."""
        cfg = EnvConfig(yield_sigma=0.10)
        rng = np.random.default_rng(0)
        base_det = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=None)
        for _ in range(200):
            y = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=rng)
            # With ±3σ=0.30 clamp: yield ∈ [base*0.7, base*1.3]
            assert y >= base_det * 0.69  # small float tolerance
            assert y <= base_det * 1.31

    def test_zero_sigma_means_deterministic(self):
        """yield_sigma=0 should produce identical results with or without rng."""
        cfg = EnvConfig(yield_sigma=0.0)
        rng = np.random.default_rng(42)
        y_no_rng = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=None)
        y_with_rng = calculate_yield(1, 4, 0.6, 0.6, False, cfg, rng=rng)
        assert y_no_rng == y_with_rng


# ── Price Autocorrelation ──────────────────────────────────────


class TestPriceAutocorrelation:
    """Tests for mean-reverting random walk market prices."""

    def test_autocorrelation_uses_prev_prices(self):
        """With autocorrelation enabled, prices should correlate with previous."""
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            price_reversion_speed=0.3,
            market_price_sigma=0.05,  # low noise for clearer signal
            demand_shock_probability=0.0,  # disable shocks for this test
        )
        rng = np.random.default_rng(42)

        # Start with very high previous prices
        high_prev = (2000.0, 1500.0, 900.0)
        prices_from_high = generate_market_prices(6, cfg, rng, prev_prices=high_prev)

        rng2 = np.random.default_rng(42)
        # Start with very low previous prices
        low_prev = (500.0, 300.0, 150.0)
        prices_from_low = generate_market_prices(6, cfg, rng2, prev_prices=low_prev)

        # Mean reversion: high prev → prices pulled down, low prev → prices pulled up
        # So prices_from_high should generally be higher than prices_from_low
        # (they track prev partially)
        assert prices_from_high[0] > prices_from_low[0]

    def test_independent_without_prev_prices(self):
        """Without prev_prices, should fall back to independent draws."""
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_market_prices(6, cfg, rng1, prev_prices=None)
        p2 = generate_market_prices(6, cfg, rng2, prev_prices=None)
        assert p1 == p2  # same seed, same result

    def test_disabled_autocorrelation(self):
        """With flag off, prev_prices should be ignored."""
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        # Pass different prev_prices — should not matter
        p1 = generate_market_prices(6, cfg, rng1, prev_prices=(2000, 2000, 2000))
        p2 = generate_market_prices(6, cfg, rng2, prev_prices=(100, 100, 100))
        assert p1 == p2

    def test_prices_stay_clamped(self):
        """Prices should never exceed base × max_multiplier."""
        cfg = EnvConfig(
            enable_price_autocorrelation=True,
            price_max_multiplier=2.5,
            demand_shock_probability=0.0,
        )
        rng = np.random.default_rng(42)
        extreme_prev = (5000.0, 5000.0, 5000.0)  # way above ceiling
        prices = generate_market_prices(6, cfg, rng, prev_prices=extreme_prev)
        for i, p in enumerate(prices):
            base = cfg.base_market_prices[i + 1]
            assert p <= base * cfg.price_max_multiplier + 0.01
            assert p >= 1.0


# ── Demand Shocks ──────────────────────────────────────────────


class TestDemandShocks:
    """Tests for rare demand shock events."""

    def test_shocks_occur_over_many_draws(self):
        """With prob=0.5, shocks should happen roughly half the time."""
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.5,
            demand_shock_magnitude=(0.3, 0.6),
        )
        # Run many draws and check if some prices deviate significantly
        rng = np.random.default_rng(42)
        base_prices = []
        for _ in range(100):
            p = generate_market_prices(6, cfg, rng)
            base_prices.append(p)

        # At least some draws should show large deviations (>15%)
        corn_prices = [p[0] for p in base_prices]
        wheat_prices = [p[1] for p in base_prices]
        chickpea_prices = [p[2] for p in base_prices]
        all_prices = corn_prices + wheat_prices + chickpea_prices

        base_corn = cfg.base_market_prices[1]
        # Some prices should deviate >20% from base (shock + noise)
        large_deviations = [
            abs(p - base_corn) / base_corn for p in corn_prices
        ]
        assert max(large_deviations) > 0.15

    def test_no_shocks_when_disabled(self):
        """With prob=0, prices should follow normal distribution only."""
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=0.0,
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        # Same seed, same result — no random shock draws consumed
        p1 = generate_market_prices(6, cfg, rng1)
        p2 = generate_market_prices(6, cfg, rng2)
        assert p1 == p2

    def test_shocked_prices_stay_clamped(self):
        """Even after a demand shock, prices should stay within bounds."""
        cfg = EnvConfig(
            enable_price_autocorrelation=False,
            demand_shock_probability=1.0,  # always shock
            demand_shock_magnitude=(0.5, 0.6),
            price_max_multiplier=2.5,
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            prices = generate_market_prices(6, cfg, rng)
            for i, p in enumerate(prices):
                base = cfg.base_market_prices[i + 1]
                assert p <= base * cfg.price_max_multiplier + 0.01
                assert p >= 1.0
