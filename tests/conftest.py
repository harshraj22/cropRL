"""Shared fixtures for CropRL tests."""

import pytest
import numpy as np

from crop_env.config import EnvConfig
from crop_env.server.crop_environment import CropEnvironment


@pytest.fixture
def config():
    """Default environment config."""
    return EnvConfig()


@pytest.fixture
def env():
    """Fresh environment, reset with seed 42."""
    e = CropEnvironment()
    e.reset(seed=42)
    return e


@pytest.fixture
def rng():
    """Seeded numpy RNG for deterministic dynamics tests."""
    return np.random.default_rng(42)
