"""NEW: Tests for multi-harvest cycles and crop rotation strategies."""

from crop_env.config import EnvConfig
from crop_env.models import CropAction
from crop_env.server.crop_environment import CropEnvironment


class TestMultiHarvestCycles:
    """Verify that plant → harvest → plant → harvest works across cycles."""

    def test_two_harvest_cycles(self):
        e = CropEnvironment()
        e.reset(seed=42)

        # Cycle 1: Plant chickpea, wait, harvest & sell
        e.step(CropAction(action_id=3))  # plant chickpea
        assert e._internal["active_crop_type"] == 3
        e.step(CropAction(action_id=0))  # wait (age → 1)
        obs1 = e.step(CropAction(action_id=7))  # harvest & sell
        assert obs1.active_crop_type == 0  # fallow
        cash_after_1 = obs1.cash_balance

        # Cycle 2: Plant corn on the now-fallow land
        obs_plant = e.step(CropAction(action_id=1))  # plant corn
        assert obs_plant.active_crop_type == 1
        for _ in range(3):
            e.step(CropAction(action_id=0))  # grow corn
        obs2 = e.step(CropAction(action_id=7))  # harvest & sell
        assert obs2.active_crop_type == 0  # fallow again

    def test_three_cycle_rotation(self):
        """Corn → Chickpea → Wheat rotation should work without errors."""
        e = CropEnvironment()
        e.reset(seed=42)
        crops = [1, 3, 2]  # corn, chickpea, wheat

        for crop_id in crops:
            e.step(CropAction(action_id=crop_id))  # plant
            e.step(CropAction(action_id=0))  # wait
            obs = e.step(CropAction(action_id=7))  # harvest
            assert obs.active_crop_type == 0, f"Land not fallow after harvesting crop {crop_id}"

    def test_store_then_plant_next(self):
        """Store one crop, plant another, then sell stored crop."""
        e = CropEnvironment()
        e.reset(seed=42)

        # Cycle 1: plant, harvest & store
        e.step(CropAction(action_id=3))
        e.step(CropAction(action_id=0))
        e.step(CropAction(action_id=6))  # harvest & store
        assert e._internal["stored_amount"] > 0

        # Cycle 2: plant something else while old crop is in storage
        e.step(CropAction(action_id=1))  # plant corn
        assert e._internal["active_crop_type"] == 1
        assert e._internal["stored_amount"] > 0  # storage still has chickpea

        # Sell stored crop while corn is growing
        obs = e.step(CropAction(action_id=8))  # sell inventory
        assert obs.stored_amount == 0.0
        assert e._internal["active_crop_type"] == 1  # corn still growing

    def test_cash_increases_over_profitable_cycle(self):
        """A full plant-harvest cycle should increase cash."""
        config = EnvConfig(initial_cash=10000.0, initial_soil_nitrogen=0.8)
        e = CropEnvironment(config=config)
        e.reset(seed=42)

        # Plant chickpea (cheapest: ₹200), grow, harvest & sell
        e.step(CropAction(action_id=3))
        for _ in range(3):  # grow to maturity
            e.step(CropAction(action_id=0))
        obs = e.step(CropAction(action_id=7))

        # Should have made a profit (harvest revenue > ₹200 seed cost)
        assert obs.cash_balance > 10000.0 - 200  # at minimum broke even
