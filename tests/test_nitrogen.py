"""Tests for soil nitrogen — recovery, depletion, fertilizer, and crop interactions."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment


class TestNitrogenRecovery:
    def test_passive_recovery_each_month(self, env):
        n_before = env._internal["soil_nitrogen"]
        env.step(CroprlAction(action_id=0))
        n_after = env._internal["soil_nitrogen"]
        assert n_after >= n_before + 0.009  # ~0.01 recovery

    def test_nitrogen_capped_at_1(self):
        config = EnvConfig(initial_soil_nitrogen=0.99)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)
        e.step(CroprlAction(action_id=5))  # fertilize +0.15
        assert e._internal["soil_nitrogen"] <= 1.0


class TestCornNitrogenDrain:
    """NEW: Corn (crop 1) should deplete soil nitrogen significantly."""

    def test_corn_depletes_nitrogen(self):
        config = EnvConfig(initial_soil_nitrogen=0.8)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)

        n_before = e._internal["soil_nitrogen"]
        e.step(CroprlAction(action_id=1))  # plant corn

        # Grow corn for a few months, then harvest
        for _ in range(3):
            e.step(CroprlAction(action_id=0))
        e.step(CroprlAction(action_id=7))  # harvest & sell

        n_after = e._internal["soil_nitrogen"]
        # Corn is a "heavy feeder" — nitrogen should drop noticeably
        # even accounting for passive recovery (+0.01/month * 4 months = +0.04)
        assert n_after < n_before

    def test_wheat_drains_less_than_corn(self):
        """Wheat (crop 2) should drain less nitrogen than corn (crop 1)."""
        # Run corn scenario
        e1 = CroprlEnvironment(config=EnvConfig(initial_soil_nitrogen=0.8))
        e1.reset(seed=42)
        e1.step(CroprlAction(action_id=1))
        for _ in range(3):
            e1.step(CroprlAction(action_id=0))
        e1.step(CroprlAction(action_id=7))
        n_after_corn = e1._internal["soil_nitrogen"]

        # Run wheat scenario
        e2 = CroprlEnvironment(config=EnvConfig(initial_soil_nitrogen=0.8))
        e2.reset(seed=42)
        e2.step(CroprlAction(action_id=2))
        for _ in range(3):
            e2.step(CroprlAction(action_id=0))
        e2.step(CroprlAction(action_id=7))
        n_after_wheat = e2._internal["soil_nitrogen"]

        assert n_after_wheat > n_after_corn


class TestChickpeaNitrogenRestore:
    """NEW: Chickpea (crop 3) should restore/maintain soil nitrogen."""

    def test_chickpea_restores_nitrogen(self):
        config = EnvConfig(initial_soil_nitrogen=0.5)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)

        n_before = e._internal["soil_nitrogen"]
        e.step(CroprlAction(action_id=3))  # plant chickpea
        for _ in range(3):
            e.step(CroprlAction(action_id=0))
        e.step(CroprlAction(action_id=7))  # harvest

        n_after = e._internal["soil_nitrogen"]
        # Chickpea is a legume with negative drain — should increase nitrogen
        assert n_after > n_before


class TestFertilizeChickpeaStacking:
    """NEW: Fertilize + chickpea should both contribute to nitrogen."""

    def test_stacking_increases_more(self):
        # Chickpea only
        e1 = CroprlEnvironment(config=EnvConfig(initial_soil_nitrogen=0.4))
        e1.reset(seed=42)
        e1.step(CroprlAction(action_id=3))  # plant chickpea
        e1.step(CroprlAction(action_id=0))
        e1.step(CroprlAction(action_id=0))
        e1.step(CroprlAction(action_id=7))  # harvest
        n_chickpea_only = e1._internal["soil_nitrogen"]

        # Chickpea + fertilize
        e2 = CroprlEnvironment(config=EnvConfig(initial_soil_nitrogen=0.4))
        e2.reset(seed=42)
        e2.step(CroprlAction(action_id=3))  # plant chickpea
        e2.step(CroprlAction(action_id=5))  # fertilize
        e2.step(CroprlAction(action_id=0))
        e2.step(CroprlAction(action_id=7))  # harvest
        n_chickpea_plus_fert = e2._internal["soil_nitrogen"]

        # Fertilize should add on top of chickpea's natural restoration
        assert n_chickpea_plus_fert > n_chickpea_only
