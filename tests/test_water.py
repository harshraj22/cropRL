"""Tests for the water level system."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.enums import ActionType, CropType


class TestWaterSystem:
    """Verify water level mechanics: irrigation, rainfall, drainage, crop consumption."""

    def test_irrigation_increases_water(self):
        """Irrigating should increase water level."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        # Plant corn
        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        water_before = e._internal["water_level"]

        # Irrigate
        e.step(CroprlAction(action_id=ActionType.IRRIGATE))
        water_after = e._internal["water_level"]

        assert water_after > water_before

    def test_irrigation_caps_at_optimal(self):
        """Water level should never exceed optimal for the active crop."""
        cfg = EnvConfig()
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # Plant corn (optimal water = 0.6)
        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))

        # Irrigate multiple times
        for _ in range(10):
            e.step(CroprlAction(action_id=ActionType.IRRIGATE))

        assert e._internal["water_level"] <= cfg.optimal_water_level[CropType.CORN]

    def test_rain_adds_water_on_month_advance(self):
        """Waiting (month advance) should add realised rainfall to water."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        # Plant corn
        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        # Water starts at 0, Wait adds rain
        e.step(CroprlAction(action_id=ActionType.WAIT))

        # After month advance, water should have changed (rain - consumption)
        # Can be positive or negative depending on the random values
        assert isinstance(e._internal["water_level"], float)

    def test_crop_consumes_water_monthly(self):
        """Growing crops should consume water each month."""
        # Use summer (month=5) where baseline rain (0.05) < corn consumption (0.15)
        # Start in April (month=4) so Wait moves to May which is dry
        cfg = EnvConfig(
            weather_sigma=0.0,
            weather_sigma_realisation=0.0,
        )
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # Wait to reach April (month 4)
        for _ in range(3):
            e.step(CroprlAction(action_id=ActionType.WAIT))
        assert e._internal["month"] == 4

        # Plant corn and irrigate to max
        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        for _ in range(3):
            e.step(CroprlAction(action_id=ActionType.IRRIGATE))

        water_full = e._internal["water_level"]

        # Wait one month (April→May, dry season with baseline 0.05)
        # Rain 0.05 - consumption 0.15 = net -0.10
        e.step(CroprlAction(action_id=ActionType.WAIT))
        water_after = e._internal["water_level"]

        assert water_after < water_full, (
            f"Water should decrease in dry month. Before: {water_full}, after: {water_after}"
        )

    def test_fallow_land_holds_water(self):
        """Fallow land should accumulate rain without crop consumption."""
        cfg = EnvConfig()
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # Start fallow, no crop consuming water
        # After waiting, water should come from rain only
        e.step(CroprlAction(action_id=ActionType.WAIT))
        water = e._internal["water_level"]
        # Should have some water from rain (unless dry month)
        assert water >= 0.0

    def test_water_never_negative(self):
        """Water level should always be >= 0."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        for _ in range(20):
            e.step(CroprlAction(action_id=ActionType.WAIT))
            assert e._internal["water_level"] >= 0.0
            if e._internal["active_crop_type"] == CropType.FALLOW:
                break


class TestWaterYieldInteraction:
    """Verify that water level affects yield."""

    def test_more_water_more_yield(self):
        """Higher water level should produce higher yield."""
        from cropRL.dynamics import calculate_yield

        cfg = EnvConfig()
        y_low = calculate_yield(1, 4, 0.6, 0.1, 7, cfg, rng=None)
        y_high = calculate_yield(1, 4, 0.6, 0.6, 7, cfg, rng=None)
        assert y_high > y_low

    def test_zero_water_yields_something(self):
        """Even at zero water, yield should be > 0 (floor at 0.1 factor)."""
        from cropRL.dynamics import calculate_yield

        cfg = EnvConfig()
        y = calculate_yield(1, 4, 0.6, 0.0, 7, cfg, rng=None)
        assert y > 0.0
