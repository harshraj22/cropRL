"""Tests for all 11 actions — valid execution and invalid rejection."""

from crop_env.models import CropAction


class TestValidActions:
    def test_wait(self, env):
        obs = env.step(CropAction(action_id=0))
        assert obs.reward is not None
        assert "waited" in obs.message.lower()

    def test_plant_crop1(self, env):
        obs = env.step(CropAction(action_id=1))
        assert obs.active_crop_type == 1
        assert obs.cash_balance < 10000.0
        assert "planted" in obs.message.lower()

    def test_plant_crop2(self, env):
        obs = env.step(CropAction(action_id=2))
        assert obs.active_crop_type == 2

    def test_plant_crop3(self, env):
        obs = env.step(CropAction(action_id=3))
        assert obs.active_crop_type == 3

    def test_irrigate(self, env):
        env.step(CropAction(action_id=1))  # plant first
        obs = env.step(CropAction(action_id=4))
        assert "irrigated" in obs.message.lower()

    def test_fertilize(self, env):
        obs_before = env.reset(seed=42)
        n_before = obs_before.soil_nitrogen
        obs = env.step(CropAction(action_id=5))
        assert obs.soil_nitrogen > n_before

    def test_harvest_and_sell(self, env):
        env.step(CropAction(action_id=3))  # plant chickpea
        env.step(CropAction(action_id=0))  # wait
        obs = env.step(CropAction(action_id=7))  # harvest & sell
        assert obs.active_crop_type == 0
        assert "harvested" in obs.message.lower()

    def test_harvest_and_store(self, env):
        env.step(CropAction(action_id=3))
        env.step(CropAction(action_id=0))
        obs = env.step(CropAction(action_id=6))
        assert obs.active_crop_type == 0
        assert obs.stored_amount > 0
        assert obs.stored_crop_type == 3

    def test_sell_inventory(self, env):
        env.step(CropAction(action_id=3))
        env.step(CropAction(action_id=0))
        env.step(CropAction(action_id=6))  # store
        obs = env.step(CropAction(action_id=8))  # sell
        assert obs.stored_amount == 0.0
        assert "sold" in obs.message.lower()

    def test_take_loan(self, env):
        cash_before = env._internal["cash"]
        obs = env.step(CropAction(action_id=9))
        assert obs.cash_balance > cash_before
        assert obs.current_debt > 0

    def test_repay_loan(self, env):
        env.step(CropAction(action_id=9))
        obs = env.step(CropAction(action_id=10))
        assert obs.current_debt == 0.0
        assert "debt-free" in obs.message.lower()


class TestInvalidActions:
    def test_plant_on_occupied_land(self, env):
        env.step(CropAction(action_id=1))
        obs = env.step(CropAction(action_id=2))
        assert "INVALID" in obs.message

    def test_irrigate_fallow(self, env):
        obs = env.step(CropAction(action_id=4))
        assert "INVALID" in obs.message

    def test_harvest_empty(self, env):
        obs = env.step(CropAction(action_id=7))
        assert "INVALID" in obs.message

    def test_sell_empty_storage(self, env):
        obs = env.step(CropAction(action_id=8))
        assert "INVALID" in obs.message

    def test_take_loan_when_active(self, env):
        env.step(CropAction(action_id=9))
        obs = env.step(CropAction(action_id=9))
        assert "INVALID" in obs.message

    def test_repay_without_loan(self, env):
        obs = env.step(CropAction(action_id=10))
        assert "INVALID" in obs.message
