"""Tests for episode termination — max steps, bankruptcy, terminal bonus."""

from crop_env.config import EnvConfig
from crop_env.models import CropAction
from crop_env.server.environment import CropEnvironment


class TestTermination:
    def test_episode_ends_at_max_steps(self):
        config = EnvConfig(max_steps=5)
        e = CropEnvironment(config=config)
        e.reset(seed=42)
        obs = None
        for _ in range(5):
            obs = e.step(CropAction(action_id=0))
        assert obs.done is True
        assert "EPISODE COMPLETE" in obs.message

    def test_terminal_bonus_includes_soil(self):
        config = EnvConfig(max_steps=1)
        e = CropEnvironment(config=config)
        e.reset(seed=42)
        obs = e.step(CropAction(action_id=0))
        assert obs.done is True
        assert obs.reward > 5000  # soil bonus dominates

    def test_full_60_step_run(self, env):
        """Smoke test: run 60 steps cycling all actions without error."""
        for i in range(60):
            action_id = i % 11
            obs = env.step(CropAction(action_id=action_id))
            if obs.done:
                break
        assert obs.done is True


class TestBankruptcy:
    """Tests for bankruptcy termination (cash < 0 with active loan)."""

    def test_bankruptcy_ends_episode(self):
        """Cash going negative with active debt should trigger bankruptcy.

        The env checks `cash < 0 and has_active_loan` after dynamics.
        Cash can't go below 0 from actions (they check balance first),
        but it CAN go negative via storage costs when enable_storage_cost=True.
        We use high storage cost + long spoilage time to ensure cash drains.
        """
        config = EnvConfig(
            initial_cash=500.0,
            enable_storage_cost=True,
            cost_storage_monthly=5000.0,  # very high: ₹5000/ton/month
            max_storage_age=60,  # won't spoil before draining cash
        )
        e = CropEnvironment(config=config)
        e.reset(seed=42)

        # Plant chickpea, wait, harvest & store
        e.step(CropAction(action_id=3))  # -₹200, cash=300
        e.step(CropAction(action_id=0))
        e.step(CropAction(action_id=6))  # harvest & store

        # Take a loan so has_active_loan=True
        e.step(CropAction(action_id=9))  # +₹5000

        # Wait — storage costs ₹5000*tons/month will drain cash negative
        obs = None
        for _ in range(10):
            obs = e.step(CropAction(action_id=0))
            if obs.done:
                break

        assert obs.done is True
        assert "BANKRUPTCY" in obs.message

    def test_no_bankruptcy_without_loan(self):
        """If there's no active loan, running out of cash doesn't end the game."""
        config = EnvConfig(initial_cash=500.0)
        e = CropEnvironment(config=config)
        e.reset(seed=42)

        # Spend most cash
        e.step(CropAction(action_id=1))  # plant corn, -₹800 → would fail
        # Actually just plant cheapest
        e2 = CropEnvironment(config=config)
        e2.reset(seed=42)
        e2.step(CropAction(action_id=3))  # plant chickpea -₹200, cash=300
        e2.step(CropAction(action_id=0))  # wait

        # Cash is low but no loan → should NOT be bankrupt
        assert e2._internal["cash"] < 500
        assert e2._internal["has_active_loan"] is False
        obs = e2.step(CropAction(action_id=0))
        assert obs.done is False
