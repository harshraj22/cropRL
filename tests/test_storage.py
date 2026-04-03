"""Tests for storage — spoilage in env and auto-sell on overwrite."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment


class TestSpoilageInEnv:
    def test_storage_rots_after_max_age(self):
        config = EnvConfig(max_storage_age=3)
        e = CroprlEnvironment(config=config)
        e.reset(seed=42)
        e.step(CroprlAction(action_id=3))  # plant chickpea
        e.step(CroprlAction(action_id=0))  # wait
        e.step(CroprlAction(action_id=6))  # harvest & store
        assert e._internal["stored_amount"] > 0
        for _ in range(4):
            e.step(CroprlAction(action_id=0))
        assert e._internal["stored_amount"] == 0.0


class TestHarvestStoreOverwrite:
    def test_auto_sells_old_storage(self, env):
        # First crop: plant, wait, store
        env.step(CroprlAction(action_id=3))
        env.step(CroprlAction(action_id=0))
        env.step(CroprlAction(action_id=6))
        assert env._internal["stored_crop_type"] == 3

        # Second crop: plant, wait, store (auto-sells old)
        env.step(CroprlAction(action_id=2))
        env.step(CroprlAction(action_id=0))
        obs = env.step(CroprlAction(action_id=6))
        assert "auto-sold" in obs.message.lower()
        assert obs.stored_crop_type == 2
