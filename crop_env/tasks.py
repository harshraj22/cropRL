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


def grader(task_id: str, final_net_worth: float, bankrupt: bool, trajectory: list[dict] = None) -> float:
    """
    Grade the agent's performance on a 0.0 - 1.0 scale.
    
    This grader uses an "Empirical Oracle Upper Bound" approach to handle stochasticity safely.
    Because weather and market prices are randomly generated, a fixed theoretical ceiling 
    punishes agents with bad RNG and breaks the 1.0 limit for lucky agents. 
    
    Instead, we reconstruct the maximum net worth specifically for the exactly dealt 
    prices the agent experienced. We calculate the absolute peak monthly profit potential 
    at each step, assuming maximum theoretical crop yields.
    
    Score = (Agent Net Worth - Baseline) / (Oracle Upper Bound - Baseline)
    
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
        
    # Minimum baseline: what the agent gets doing literally nothing.
    baseline_min = 10000.0  
    
    if final_net_worth <= baseline_min:
        return 0.0

    # Calculate Oracle Upper Bound from strictly this episode's market prices
    # Maximum possible nitrogen factor is mathematically floored at 1.5x of base yields
    # Corn base 8.0 * 1.5 = 12.0 tons
    # Wheat base 5.0 * 1.5 = 7.5 tons
    # Chickpea base 3.0 * 1.5 = 4.5 tons
    total_oracle_profit = 0.0
    
    for step_data in trajectory:
        # Extract the prices logged at this step (fallback to base configs if malformed)
        prices = step_data.get("prices", [1200.0, 800.0, 500.0])
        
        # Monthly amortized profit potentials: (Price * MaxYield - SeedCost) / GrowthMonths
        corn_prof = ((prices[0] * 12.0) - 800.0) / 4.0
        wheat_prof = ((prices[1] * 7.5) - 500.0) / 3.0
        chickpea_prof = ((prices[2] * 4.5) - 200.0) / 3.0
        
        # A theoretical oracle farmer gets the most profitable crop's value this month
        total_oracle_profit += max(0.0, corn_prof, wheat_prof, chickpea_prof)
        
    # Max possible net worth = Starting Cash + Max Perfect Soil Bonus + Oracle Farm Profit
    oracle_max_net_worth = 10000.0 + 10000.0 + total_oracle_profit
    
    # Standard min-max normalization
    score = (final_net_worth - baseline_min) / (oracle_max_net_worth - baseline_min)
    
    # Clamp mathematically guarantees we never violate hackathon bounds
    return float(max(0.0, min(1.0, score)))
