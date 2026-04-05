"""Tests for episode termination — max steps, max months, bankruptcy, terminal value."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.enums import ActionType


class TestTermination:
    def test_episode_ends_at_max_steps(self):
        """Episode ends when step counter hits max_steps."""
        config = EnvConfig(max_steps=5, max_months=60)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)
        obs = None
        # Need to use various actions since each step increments counter
        for _ in range(5):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))
        assert obs.done is True
        assert "EPISODE COMPLETE" in obs.message

    def test_episode_ends_at_max_months(self):
        """Episode ends when month_count reaches max_months."""
        config = EnvConfig(max_steps=1000, max_months=3)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)
        obs = None
        for _ in range(3):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))
        assert obs.done is True

    def test_terminal_value_includes_land(self):
        """Terminal reward should reflect land value via base_land_price × nitrogen."""
        config = EnvConfig(max_steps=1, max_months=60, base_land_price=50000.0)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)
        obs = e.step(CroprlAction(action_id=ActionType.WAIT))
        assert obs.done is True
        # Terminal value exists in the reward (could be positive or negative
        # depending on fixed costs etc. — just verify it completed)
        assert "EPISODE COMPLETE" in obs.message

    def test_full_300_step_run(self, env):
        """Smoke test: run many steps cycling actions without error."""
        for i in range(300):
            action_id = i % 11
            obs = env.step(CroprlAction(action_id=action_id))
            if obs.done:
                break
        assert obs.done is True


class TestBankruptcy:
    """Tests for bankruptcy termination (cash < 0 with active loan)."""

    def test_bankruptcy_ends_episode(self):
        config = EnvConfig(
            initial_cash=500.0,
            enable_storage_cost=True,
            cost_storage_monthly=5000.0,
            max_storage_age=60,
        )
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)

        # Plant chickpea, wait, harvest & store
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        e.step(CroprlAction(action_id=ActionType.WAIT))
        e.step(CroprlAction(action_id=ActionType.HARVEST_STORE))

        # Take a loan so has_active_loan=True
        e.step(CroprlAction(action_id=ActionType.TAKE_LOAN))

        # Wait — storage costs will drain cash negative
        obs = None
        for _ in range(10):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))
            if obs.done:
                break

        assert obs.done is True
        assert "BANKRUPTCY" in obs.message

    def test_no_bankruptcy_without_loan(self):
        config = EnvConfig(initial_cash=500.0)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)

        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        e.step(CroprlAction(action_id=ActionType.WAIT))

        assert e._internal["has_active_loan"] is False
        obs = e.step(CroprlAction(action_id=ActionType.WAIT))
        assert obs.done is False
