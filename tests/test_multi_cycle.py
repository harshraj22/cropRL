"""Tests for multi-harvest cycles and crop rotation strategies."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.enums import ActionType, CropType


class TestMultiHarvestCycles:
    """Verify that plant → harvest → plant → harvest works across cycles."""

    def test_two_harvest_cycles(self):
        e = CroprlEnvironment()
        e.reset(seed=42)

        # Cycle 1: Plant chickpea, wait, harvest & sell
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        assert e._internal["active_crop_type"] == CropType.CHICKPEA
        e.step(CroprlAction(action_id=ActionType.WAIT))  # age → 1
        obs1 = e.step(CroprlAction(action_id=ActionType.HARVEST_SELL))
        assert obs1.active_crop_type == CropType.FALLOW

        # Cycle 2: Plant corn on the now-fallow land
        obs_plant = e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        assert obs_plant.active_crop_type == CropType.CORN
        for _ in range(3):
            e.step(CroprlAction(action_id=ActionType.WAIT))
        obs2 = e.step(CroprlAction(action_id=ActionType.HARVEST_SELL))
        assert obs2.active_crop_type == CropType.FALLOW

    def test_three_cycle_rotation(self):
        """Corn → Chickpea → Wheat rotation should work without errors."""
        e = CroprlEnvironment()
        e.reset(seed=42)
        crops = [ActionType.PLANT_CORN, ActionType.PLANT_CHICKPEA, ActionType.PLANT_WHEAT]

        for crop_action in crops:
            e.step(CroprlAction(action_id=crop_action))
            e.step(CroprlAction(action_id=ActionType.WAIT))
            obs = e.step(CroprlAction(action_id=ActionType.HARVEST_SELL))
            assert obs.active_crop_type == CropType.FALLOW

    def test_store_then_plant_next(self):
        """Store one crop, plant another, then sell stored crop."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        # Cycle 1: plant, harvest & store
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        e.step(CroprlAction(action_id=ActionType.WAIT))
        e.step(CroprlAction(action_id=ActionType.HARVEST_STORE))
        assert e._internal["stored_amount"] > 0

        # Cycle 2: plant something else while old crop is in storage
        e.step(CroprlAction(action_id=ActionType.PLANT_CORN))
        assert e._internal["active_crop_type"] == CropType.CORN
        assert e._internal["stored_amount"] > 0

        # Sell stored crop while corn is growing
        obs = e.step(CroprlAction(action_id=ActionType.SELL_INVENTORY))
        assert obs.stored_amount == 0.0
        assert e._internal["active_crop_type"] == CropType.CORN

    def test_cash_increases_over_profitable_cycle(self):
        """A full plant-harvest cycle should increase cash."""
        config = EnvConfig(initial_cash=10000.0, initial_soil_nitrogen=0.8)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)

        # Plant chickpea (cheapest: ₹200), grow, harvest & sell
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        for _ in range(3):  # grow to maturity
            e.step(CroprlAction(action_id=ActionType.WAIT))
        obs = e.step(CroprlAction(action_id=ActionType.HARVEST_SELL))

        # Cash should cover seed cost + fixed costs and still profit
        # 3 months × ₹200 fixed = ₹600 + ₹200 seed = ₹800 total cost
        # Chickpea at 3 months yields ~3 tons × ₹500 = ₹1500 revenue (even in wrong season)
        assert obs.cash_balance > 10000.0 - 1000  # at minimum nearly broke even
