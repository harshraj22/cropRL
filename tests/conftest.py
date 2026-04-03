"""Shared fixtures for CropRL tests."""

import pytest
import numpy as np

from cropRL.config import EnvConfig
from cropRL.server.cropRL_environment import CroprlEnvironment


@pytest.fixture
def config():
    """Default environment config."""
    return EnvConfig()


@pytest.fixture
def env():
    """Fresh environment, reset with seed 42."""
    e = CroprlEnvironment()
    e.reset(seed=42)
    return e


@pytest.fixture
def rng():
    """Seeded numpy RNG for deterministic dynamics tests."""
    return np.random.default_rng(42)
