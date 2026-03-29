"""
CropRL Task Definitions.

Three tasks with the same objective (maximize net worth over 60 months)
but different environment complexity levels.
"""

from __future__ import annotations

from .config import EnvConfig
from .server.crop_environment import CropEnvironment


TASKS: dict[str, dict] = {
    "easy": {
        "description": (
            "Maximize net worth over 60 months. Simplified conditions: "
            "no interest on loans, stable weather, generous starting capital."
        ),
        "config_overrides": {
            "initial_cash": 15000.0,
            "base_interest_rate": 0.0,       # No loan interest
            "weather_sigma": 0.05,            # Very predictable weather
            "market_price_sigma": 0.05,       # Stable markets
            "initial_soil_nitrogen": 0.8,     # Healthy soil
            "max_storage_age": 12,            # Slow spoilage
        },
    },
    "medium": {
        "description": (
            "Maximize net worth over 60 months. Standard conditions."
        ),
        "config_overrides": {
            # All defaults from EnvConfig
        },
    },
    "hard": {
        "description": (
            "Maximize net worth over 60 months. Harsh conditions: "
            "high interest, volatile weather and markets, poor starting soil."
        ),
        "config_overrides": {
            "initial_cash": 7000.0,
            "base_interest_rate": 0.12,       # 12% interest
            "weather_sigma": 0.25,            # Unpredictable weather
            "market_price_sigma": 0.20,       # Volatile markets
            "initial_soil_nitrogen": 0.35,    # Degraded soil
            "max_storage_age": 4,             # Fast spoilage
        },
    },
}


def create_env_for_task(
    task_id: str, text_mode: bool = False
) -> CropEnvironment:
    """
    Create a CropEnvironment configured for the given task.

    Parameters
    ----------
    task_id : str
        One of "easy", "medium", "hard".
    text_mode : bool
        Whether to enable text observation mode (for LLM agents).

    Returns
    -------
    CropEnvironment
        Environment with task-specific configuration.

    Raises
    ------
    KeyError
        If task_id is not recognised.
    """
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
        )

    overrides = TASKS[task_id]["config_overrides"].copy()
    overrides["text_mode"] = text_mode
    config = EnvConfig(**overrides)
    return CropEnvironment(config=config, task_id=task_id)


def list_tasks() -> dict[str, str]:
    """Return a dict of task_id → description."""
    return {tid: info["description"] for tid, info in TASKS.items()}
