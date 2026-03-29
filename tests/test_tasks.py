"""Tests for task difficulty config overrides."""

import pytest

from crop_env.config import EnvConfig
from crop_env.tasks import create_env_for_task


class TestTasks:
    def test_easy_has_zero_interest(self):
        e = create_env_for_task("easy")
        assert e.config.base_interest_rate == 0.0

    def test_easy_has_more_cash(self):
        e = create_env_for_task("easy")
        assert e.config.initial_cash == 15000.0

    def test_hard_has_less_cash(self):
        e = create_env_for_task("hard")
        assert e.config.initial_cash == 7000.0

    def test_hard_has_higher_interest(self):
        e = create_env_for_task("hard")
        assert e.config.base_interest_rate == 0.12

    def test_medium_is_default(self):
        e = create_env_for_task("medium")
        default = EnvConfig()
        assert e.config.initial_cash == default.initial_cash
        assert e.config.base_interest_rate == default.base_interest_rate

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            create_env_for_task("nonexistent")

    def test_text_mode_passthrough(self):
        e = create_env_for_task("easy", text_mode=True)
        assert e.config.text_mode is True
