"""Tests for loan mechanics — limits, interest, repayment."""

from crop_env.config import EnvConfig
from crop_env.models import CropAction
from crop_env.server.crop_environment import CropEnvironment


class TestLoanMechanics:
    def test_only_one_loan(self, env):
        env.step(CropAction(action_id=9))
        assert env._internal["has_active_loan"] is True
        obs = env.step(CropAction(action_id=9))
        assert "INVALID" in obs.message

    def test_repay_full_amount(self, env):
        env.step(CropAction(action_id=9))
        env.step(CropAction(action_id=0))  # interest accumulates
        debt_before = env._internal["debt"]
        assert debt_before > 5000.0
        obs = env.step(CropAction(action_id=10))
        assert obs.current_debt == 0.0
        assert env._internal["has_active_loan"] is False

    def test_cannot_repay_insufficient_cash(self):
        config = EnvConfig(initial_cash=3000.0)
        e = CropEnvironment(config=config)
        e.reset(seed=42)
        e.step(CropAction(action_id=9))  # cash=8000, debt=5000
        for _ in range(10):
            e.step(CropAction(action_id=5))  # fertilize costs ₹400 each
        obs = e.step(CropAction(action_id=10))
        assert "INVALID" in obs.message

    def test_interest_accumulates(self, env):
        env.step(CropAction(action_id=9))
        initial_debt = 5000.0
        for _ in range(3):
            env.step(CropAction(action_id=0))
        assert env._internal["debt"] > initial_debt
