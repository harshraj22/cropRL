"""Tests for the multi-agent CropRL environment."""
import pytest
from cropRL.config import EnvConfig
from cropRL.multi_env import MultiAgentCroprlEnv
from cropRL.models import CroprlAction
from cropRL.enums import ActionType, CropType
from cropRL.dynamics import calculate_supply_adjusted_price

@pytest.fixture
def env2():
    cfg = EnvConfig(num_farmers=2, steps_per_agent_per_month=5,
        weather_sigma=0.0, weather_sigma_realisation=0.0, market_price_sigma=0.0,
        yield_sigma=0.0, demand_shock_probability=0.0, enable_price_autocorrelation=False)
    env = MultiAgentCroprlEnv(config=cfg, task_id="test")
    env.reset(seed=42)
    return env

class TestSupplyPricing:
    def test_single_seller(self):
        assert calculate_supply_adjusted_price(1000.0, 1, 0.15) == 1000.0
    def test_two_sellers(self):
        assert abs(calculate_supply_adjusted_price(1000.0, 2, 0.15) - 1000.0/1.15) < 0.01

class TestReset:
    def test_returns_all_farmers(self, env2):
        obs = env2.reset(seed=1)
        assert len(obs) == 2 and 0 in obs and 1 in obs
    def test_shared_weather(self, env2):
        assert env2.obs(0).expected_rainfall == env2.obs(1).expected_rainfall

class TestTwoPhase:
    def test_immediate_executes(self, env2):
        r = env2.step(CroprlAction(action_id=ActionType.FERTILIZE, farmer_id=0), 0, end=False)
        assert r.soil_nitrogen > 0.6
    def test_deferred_queued(self, env2):
        r = env2.step(CroprlAction(action_id=ActionType.PLANT_CORN, farmer_id=0), 0, end=False)
        assert "Queued" in r.message and r.active_crop_type == CropType.FALLOW
    def test_deferred_resolves(self, env2):
        env2.step(CroprlAction(action_id=ActionType.PLANT_CORN, farmer_id=0), 0, end=False)
        for _ in range(4): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=1), 1, end=False)
        r = env2.step(None, 0, end=True)
        assert r.active_crop_type == CropType.CORN
    def test_substep_limit(self, env2):
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        r = env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        assert r.reward < 0

class TestForum:
    def test_visible(self, env2):
        env2.step(CroprlAction(action_id=ActionType.POST_FORUM, farmer_id=0, forum_message="grow wheat!"), 0, end=False)
        assert len(env2.obs(1).forum_messages) == 1 and "wheat" in env2.obs(1).forum_messages[0].lower()
    def test_clears_on_advance(self, env2):
        env2.step(CroprlAction(action_id=ActionType.POST_FORUM, farmer_id=0, forum_message="Hi"), 0, end=False)
        for _ in range(4): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=1), 1, end=False)
        env2.step(None, 0, end=True); env2.step(None, 1, end=True)
        assert len(env2.obs(0).forum_messages) == 0

class TestVisibility:
    def test_fallow_hidden(self, env2):
        assert env2.obs(0).other_farmers_crops == []
    def test_non_fallow_visible(self, env2):
        env2.step(CroprlAction(action_id=ActionType.PLANT_CORN, farmer_id=0), 0, end=False)
        for _ in range(4): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=1), 1, end=False)
        env2.step(None, 0, end=True); env2.step(None, 1, end=True)
        obs0 = env2.obs(0)
        assert obs0.active_crop_type == CropType.CORN and len(obs0.other_farmers_crops) == 1

class TestMonthAdvance:
    def test_advances_after_all_phase2(self, env2):
        m0 = env2.obs(0).current_month
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=0), 0, end=False)
        for _ in range(5): env2.step(CroprlAction(action_id=ActionType.NO_OP, farmer_id=1), 1, end=False)
        env2.step(None, 0, end=True)
        assert env2._shared["month"] == m0  # not yet
        env2.step(None, 1, end=True)
        assert env2._shared["month"] == (m0 % 12) + 1  # now
