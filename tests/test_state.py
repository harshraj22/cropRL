"""Tests for the OpenEnv State interface."""

from crop_env.models import CropAction
from crop_env.server.environment import CropEnvironment


class TestState:
    def test_state_step_count(self, env):
        assert env.state.step_count == 0
        env.step(CropAction(action_id=0))
        assert env.state.step_count == 1

    def test_state_episode_id(self, env):
        assert env.state.episode_id is not None
        assert len(env.state.episode_id) > 0

    def test_state_task_id(self):
        e = CropEnvironment(task_id="hard")
        e.reset(seed=42)
        assert e.state.task_id == "hard"
