"""Tests for step/month separation — the key architectural change in v2."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.enums import ActionType


class TestStepMonthSeparation:
    """Only Wait should advance the calendar month."""

    def test_wait_advances_month(self):
        """Wait action should increment month_count and change calendar month."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        assert e._internal["month"] == 1
        assert e._internal["month_count"] == 0

        e.step(CroprlAction(action_id=ActionType.WAIT))

        assert e._internal["month"] == 2
        assert e._internal["month_count"] == 1

    def test_non_wait_does_not_advance_month(self):
        """Non-Wait actions should NOT change the month."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        assert e._internal["month"] == 1

        # Plant chickpea (instant action)
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))

        assert e._internal["month"] == 1  # still January
        assert e._internal["month_count"] == 0

    def test_step_always_increments(self):
        """Step counter should increment on every action (Wait or not)."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        assert e._internal["step"] == 0

        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        assert e._internal["step"] == 1

        e.step(CroprlAction(action_id=ActionType.FERTILIZE))
        assert e._internal["step"] == 2

        e.step(CroprlAction(action_id=ActionType.WAIT))
        assert e._internal["step"] == 3

    def test_multiple_actions_per_month(self):
        """Agent should be able to take multiple actions in a single month."""
        e = CroprlEnvironment()
        e.reset(seed=42)

        # All in January
        e.step(CroprlAction(action_id=ActionType.PLANT_CHICKPEA))
        e.step(CroprlAction(action_id=ActionType.IRRIGATE))
        e.step(CroprlAction(action_id=ActionType.FERTILIZE))

        assert e._internal["month"] == 1  # still January
        assert e._internal["step"] == 3  # 3 steps taken

        # Now advance month
        e.step(CroprlAction(action_id=ActionType.WAIT))
        assert e._internal["month"] == 2  # February
        assert e._internal["step"] == 4

    def test_month_wraps_correctly(self):
        """Month should wrap from 12 → 1 (December → January)."""
        cfg = EnvConfig(max_months=60, max_steps=300)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # Wait 12 months to wrap through one year
        for _ in range(12):
            e.step(CroprlAction(action_id=ActionType.WAIT))

        assert e._internal["month"] == 1  # back to January
        assert e._internal["month_count"] == 12


class TestStepVsMonthTermination:
    """Test that both step limit and month limit can trigger termination."""

    def test_step_limit_first(self):
        """If max_steps is small, episode ends by step limit."""
        cfg = EnvConfig(max_steps=3, max_months=60)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        for _ in range(3):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))

        assert obs.done is True

    def test_month_limit_first(self):
        """If max_months is small, episode ends by month limit."""
        cfg = EnvConfig(max_steps=1000, max_months=2)
        e = CroprlEnvironment(config=cfg)
        e.reset(seed=42)

        # 2 waits = 2 months
        for _ in range(2):
            obs = e.step(CroprlAction(action_id=ActionType.WAIT))

        assert obs.done is True
