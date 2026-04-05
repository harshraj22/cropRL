"""
CropRL Task Definitions.

Three tasks with the same objective (maximize net worth over 60 months)
but different environment complexity levels.
"""

from __future__ import annotations

from .config import EnvConfig
from .server.cropRL_environment import CroprlEnvironment


TASKS: dict[str, dict] = {
    "easy": {
        "description": (
            "Maximize net worth over 60 months. Simplified conditions: "
            "no interest on loans, stable weather, generous starting capital, "
            "no inflation."
        ),
        "config_overrides": {
            "initial_cash": 15000.0,
            "base_interest_rate": 0.0,
            "weather_sigma": 0.05,
            "weather_sigma_realisation": 0.02,
            "market_price_sigma": 0.05,
            "initial_soil_nitrogen": 0.8,
            "max_storage_age": 12,
            "inflation_rate": 0.0,
            "monthly_fixed_cost": 100.0,
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
            "high interest, volatile weather and markets, poor starting soil, "
            "high inflation."
        ),
        "config_overrides": {
            "initial_cash": 7000.0,
            "base_interest_rate": 0.12,
            "weather_sigma": 0.25,
            "weather_sigma_realisation": 0.08,
            "market_price_sigma": 0.20,
            "initial_soil_nitrogen": 0.35,
            "max_storage_age": 4,
            "inflation_rate": 0.07,
            "monthly_fixed_cost": 300.0,
        },
    },
}


def create_env_for_task(
    task_id: str, text_mode: bool = False
) -> CroprlEnvironment:
    """
    Create a CroprlEnvironment configured for the given task.

    Parameters
    ----------
    task_id : str
        One of "easy", "medium", "hard".
    text_mode : bool
        Whether to enable text observation mode (for LLM agents).

    Returns
    -------
    CroprlEnvironment

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
    return CroprlEnvironment(config=config, task_id=task_id)


def list_tasks() -> dict[str, str]:
    """Return a dict of task_id → description."""
    return {tid: info["description"] for tid, info in TASKS.items()}


def grader(
    task_id: str,
    final_net_worth: float,
    bankrupt: bool,
    trajectory: list[dict] | None = None,
) -> float:
    """
    Grade the agent's performance on a 0.0 – 1.0 scale.

    Uses an "Empirical Oracle Upper Bound" approach.  The oracle
    calculates the theoretical maximum monthly profit from the exact
    market prices the agent experienced, using the new 4-factor yield
    formula at peak conditions.

    Score = (Agent Net Worth − Baseline) / (Oracle Upper Bound − Baseline)

    Parameters
    ----------
    task_id : str
        The task that was executed ("easy", "medium", "hard").
    final_net_worth : float
        The agent's net worth at the end of the episode.
    bankrupt : bool
        Whether the agent went bankrupt.
    trajectory : list[dict]
        Chronological list of step data containing exact market prices.

    Returns
    -------
    float
        A score clamped between 0.0 and 1.0.
    """
    if bankrupt or final_net_worth <= 0 or not trajectory:
        return 0.0

    # Reconstruct config for this task to get initial values
    overrides = TASKS.get(task_id, {}).get("config_overrides", {})
    cfg = EnvConfig(**overrides)

    # Baseline: net worth if agent does absolutely nothing for 60 months
    # Cash stays, land value stays, no crops, no debt
    # But monthly fixed cost erodes cash: 60 months × fixed_cost
    baseline_cash = cfg.initial_cash - (cfg.max_months * cfg.monthly_fixed_cost)
    baseline_land = cfg.base_land_price * cfg.initial_soil_nitrogen
    baseline_min = baseline_cash + baseline_land

    if final_net_worth <= baseline_min:
        return 0.0

    # Oracle Upper Bound: maximum possible profit from these prices
    # Assumes perfect nitrogen (factor=1.0), full water (factor=1.0),
    # optimal season (factor=1.0), peak maturity (factor=1.0)
    # → yield = base_yield_tons[crop] at maximum
    total_oracle_profit = 0.0

    for step_data in trajectory:
        prices = step_data.get("prices", [1200.0, 800.0, 500.0])

        # Monthly amortized profit: (price × max_yield − seed_cost) / growth_months
        corn_prof = ((prices[0] * 8.0) - 800.0) / 4.0
        wheat_prof = ((prices[1] * 5.0) - 500.0) / 3.0
        chickpea_prof = ((prices[2] * 3.0) - 200.0) / 3.0

        total_oracle_profit += max(0.0, corn_prof, wheat_prof, chickpea_prof)

    # Oracle max net worth includes perfect soil maintenance
    oracle_land = cfg.base_land_price * 1.0  # perfect nitrogen
    oracle_max = cfg.initial_cash + oracle_land + total_oracle_profit

    if oracle_max <= baseline_min:
        return 0.5  # edge case: prices so bad oracle can't beat baseline

    score = (final_net_worth - baseline_min) / (oracle_max - baseline_min)
    return float(max(0.0, min(1.0, score)))
