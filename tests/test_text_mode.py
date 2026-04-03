"""Tests for text observation mode (LLM-friendly output)."""

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment


class TestTextMode:
    def test_text_mode_off_no_summary(self, env):
        obs = env.step(CroprlAction(action_id=0))
        assert obs.text_summary == ""

    def test_text_mode_on_has_summary(self):
        config = EnvConfig(text_mode=True)
        e = CroprlEnvironment(config=config)
        obs = e.reset(seed=42)
        assert obs.text_summary != ""
        assert "Farm Dashboard" in obs.text_summary

    def test_text_summary_contains_key_info(self):
        config = EnvConfig(text_mode=True)
        e = CroprlEnvironment(config=config)
        obs = e.reset(seed=42)
        assert "January" in obs.text_summary
        assert "Soil Nitrogen" in obs.text_summary
        assert "Cash" in obs.text_summary
