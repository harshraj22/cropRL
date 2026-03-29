"""Tests for environment reset and initial state."""

from crop_env.server.crop_environment import CropEnvironment


class TestReset:
    def test_initial_cash(self, env):
        obs = env.reset(seed=42)
        assert obs.cash_balance == 10000.0

    def test_initial_state_clean(self, env):
        obs = env.reset(seed=42)
        assert obs.active_crop_type == 0
        assert obs.crop_age_months == 0
        assert obs.stored_amount == 0.0
        assert obs.current_debt == 0.0
        assert obs.current_step == 0
        assert obs.current_month == 1
        assert obs.done is False

    def test_initial_soil_nitrogen(self, env):
        obs = env.reset(seed=42)
        assert obs.soil_nitrogen == 0.6

    def test_reproducibility_with_seed(self):
        e1 = CropEnvironment()
        obs1 = e1.reset(seed=123)

        e2 = CropEnvironment()
        obs2 = e2.reset(seed=123)

        assert obs1.expected_rainfall == obs2.expected_rainfall
        assert obs1.market_price_crop_1 == obs2.market_price_crop_1

    def test_different_seeds_different_values(self):
        e1 = CropEnvironment()
        obs1 = e1.reset(seed=1)

        e2 = CropEnvironment()
        obs2 = e2.reset(seed=999)

        assert (
            obs1.expected_rainfall != obs2.expected_rainfall
            or obs1.market_price_crop_1 != obs2.market_price_crop_1
        )
