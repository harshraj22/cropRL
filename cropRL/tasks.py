"""
CropRL Task Definitions.

Three single-agent tasks with the same objective (maximize net worth over
60 months) at different difficulty levels.

Multi-agent task variants are also provided:
  easy_2agent, easy_4agent, easy_8agent
  medium_2agent, medium_4agent, medium_8agent
  hard_2agent, hard_4agent, hard_8agent
"""

from __future__ import annotations

from typing import Optional

from .config import EnvConfig, MultiAgentConfig


# ── Single-agent tasks ──────────────────────────────────────────────────────

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

# ── Multi-agent task variants ───────────────────────────────────────────────
# Injected after TASKS is defined so config_overrides references are valid.

for _difficulty in ("easy", "medium", "hard"):
    for _n in (2, 4, 8):
        _key = f"{_difficulty}_{_n}agent"
        TASKS[_key] = {
            "description": (
                f"Multi-agent ({_n} farms) variant of the '{_difficulty}' task. "
                "Agents compete on a shared market with supply-demand pricing "
                "and hype crop cycles."
            ),
            "config_overrides": TASKS[_difficulty]["config_overrides"],
            "multi_agent": True,
            "num_agents": _n,
        }


# ── Environment factories ───────────────────────────────────────────────────


def create_env_for_task(
    task_id: str,
    text_mode: bool = False,
    objective_mode: str = "competitive",
) -> "MultiAgentCroprlEnvironment":
    """
    Create a MultiAgentCroprlEnvironment configured for the given task.

    For single-agent tasks (``"easy"``, ``"medium"``, ``"hard"``), the
    environment is created with ``num_agents=1``.

    For multi-agent tasks (``"easy_4agent"``, etc.), the agent count is
    read from the task definition.

    Parameters
    ----------
    task_id : str
        Any recognised task id (single-agent or multi-agent).
    text_mode : bool
        Enable text observation mode (for LLM agents).
    objective_mode : str
        ``"competitive"``, ``"cooperative"``, or ``"mixed"``.

    Returns
    -------
    MultiAgentCroprlEnvironment
    """
    from .multi_agent_environment import MultiAgentCroprlEnvironment

    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
        )

    task_info = TASKS[task_id]
    num_agents = task_info.get("num_agents", 1)
    overrides = task_info["config_overrides"].copy()
    overrides["text_mode"] = text_mode
    env_cfg = EnvConfig(**overrides)
    ma_cfg = MultiAgentConfig(
        num_agents=num_agents,
        objective_mode=objective_mode,
    )
    return MultiAgentCroprlEnvironment(
        env_config=env_cfg,
        ma_config=ma_cfg,
        task_id=task_id,
    )


def create_multi_agent_env_for_task(
    task_id: str,
    text_mode: bool = False,
    objective_mode: str = "competitive",
) -> "MultiAgentCroprlEnvironment":
    """
    Create a MultiAgentCroprlEnvironment configured for the given multi-agent task.

    Parameters
    ----------
    task_id : str
        A multi-agent task id, e.g. ``"easy_4agent"`` or ``"hard_8agent"``.
    text_mode : bool
        Enable text observation mode (for LLM agents).
    objective_mode : str
        ``"competitive"``, ``"cooperative"``, or ``"mixed"``.

    Returns
    -------
    MultiAgentCroprlEnvironment
    """
    from .multi_agent_environment import MultiAgentCroprlEnvironment

    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}"
        )

    task_info = TASKS[task_id]
    num_agents = task_info.get("num_agents", 4)
    overrides = task_info["config_overrides"].copy()
    overrides["text_mode"] = text_mode
    env_cfg = EnvConfig(**overrides)
    ma_cfg = MultiAgentConfig(
        num_agents=num_agents,
        objective_mode=objective_mode,
    )
    return MultiAgentCroprlEnvironment(
        env_config=env_cfg,
        ma_config=ma_cfg,
        task_id=task_id,
    )


def list_tasks() -> dict[str, str]:
    """Return a dict of task_id → description."""
    return {tid: info["description"] for tid, info in TASKS.items()}


# ── Grading ─────────────────────────────────────────────────────────────────


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
        The task that was executed (``"easy"``, ``"medium"``, ``"hard"``).
        Multi-agent task ids (e.g. ``"medium_4agent"``) are also accepted;
        the base difficulty will be extracted automatically.
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
        return 0.01

    # Resolve base task (strip multi-agent suffix like "_4agent")
    base_task = task_id
    for _n in (2, 4, 8):
        base_task = base_task.replace(f"_{_n}agent", "")
    base_task = base_task if base_task in ("easy", "medium", "hard") else "medium"

    # Reconstruct config for this task to get initial values
    overrides = TASKS.get(base_task, {}).get("config_overrides", {})
    cfg = EnvConfig(**overrides)

    # Baseline: net worth if agent does absolutely nothing for 60 months
    baseline_cash = cfg.initial_cash - (cfg.max_months * cfg.monthly_fixed_cost)
    baseline_land = cfg.base_land_price * cfg.initial_soil_nitrogen
    baseline_min = baseline_cash + baseline_land

    if final_net_worth <= baseline_min:
        return 0.01

    # Oracle Upper Bound: maximum possible profit from these prices
    # Assumes perfect conditions (nitrogen=1, water=1, optimal season, peak maturity)
    total_oracle_profit = 0.0

    for step_data in trajectory:
        prices = step_data.get("prices", list(cfg.base_market_prices[1:]))

        # Monthly amortized profit: (price × max_yield − seed_cost) / growth_months
        corn_prof = ((prices[0] * cfg.base_yield_tons[1]) - cfg.seed_costs[1]) / float(cfg.growth_months[1])
        wheat_prof = ((prices[1] * cfg.base_yield_tons[2]) - cfg.seed_costs[2]) / float(cfg.growth_months[2])
        chickpea_prof = ((prices[2] * cfg.base_yield_tons[3]) - cfg.seed_costs[3]) / float(cfg.growth_months[3])

        total_oracle_profit += max(0.0, corn_prof, wheat_prof, chickpea_prof)

    # Oracle max net worth includes perfect soil maintenance
    oracle_land = cfg.base_land_price * 1.0  # perfect nitrogen
    oracle_max = cfg.initial_cash + oracle_land + total_oracle_profit

    if oracle_max <= baseline_min:
        return 0.5  # edge case: prices so bad oracle can't beat baseline

    score = (final_net_worth - baseline_min) / (oracle_max - baseline_min)
    return float(max(0.01, min(0.99, score)))


# ── Single-agent grader classes ─────────────────────────────────────────────


class EasyGrader:
    def grade(self, final_net_worth: float, bankrupt: bool, trajectory: list[dict] | None = None) -> float:
        return max(0.01, min(0.99, grader("easy", final_net_worth, bankrupt, trajectory)))


class MediumGrader:
    def grade(self, final_net_worth: float, bankrupt: bool, trajectory: list[dict] | None = None) -> float:
        return max(0.01, min(0.99, grader("medium", final_net_worth, bankrupt, trajectory)))


class HardGrader:
    def grade(self, final_net_worth: float, bankrupt: bool, trajectory: list[dict] | None = None) -> float:
        return max(0.01, min(0.99, grader("hard", final_net_worth, bankrupt, trajectory)))


# ── Multi-agent grader ──────────────────────────────────────────────────────


class MultiAgentGrader:
    """
    Grader for multi-agent episodes.

    Delegates per-agent scoring to the single-agent ``grader()`` and
    aggregates the result via ``MultiAgentCroprlEnvironment.compute_result()``.
    """

    def __init__(self, task_id: str = "medium_4agent") -> None:
        self.task_id = task_id

    def grade(
        self,
        env: "MultiAgentCroprlEnvironment",  # noqa: F821
        trajectories: Optional[dict] = None,
    ) -> "MultiAgentResult":  # noqa: F821
        """Return a MultiAgentResult for a completed episode."""
        return env.compute_result(trajectories)


# ── Rule-based agent helper ─────────────────────────────────────────────────


def _rule_based_action(obs: "MultiAgentObservation") -> int:  # noqa: F821
    """
    Simple deterministic agent used for smoke-testing multi-agent episodes.
    Priority: harvest if mature → plant if fallow → irrigate → wait.
    """
    from .enums import ActionType, CropType

    s_crop = obs.active_crop_type
    s_age = obs.crop_age_months
    s_water = obs.current_water_level

    # Harvest if crop is mature or rotting
    if s_crop != CropType.FALLOW and s_age >= 3:
        return ActionType.HARVEST_SELL

    # Plant corn if fallow and have cash
    if s_crop == CropType.FALLOW and obs.cash_balance >= obs.cost_seed_1:
        return ActionType.PLANT_CORN

    # Irrigate if water is low
    if (s_crop != CropType.FALLOW
            and s_water < 0.3
            and obs.cash_balance >= obs.cost_irrigate):
        return ActionType.IRRIGATE

    return ActionType.WAIT


# ── Multi-agent episode runner ──────────────────────────────────────────────


def run_multi_agent_episode(
    task_id: str = "medium_4agent",
    num_agents: Optional[int] = None,
    seed: int = 42,
    agent_fn=None,
    verbose: bool = False,
) -> "MultiAgentResult":  # noqa: F821
    """
    Run a complete multi-agent episode with rule-based (or custom) agents.

    Parameters
    ----------
    task_id : str
        Multi-agent task identifier, e.g. ``"easy_4agent"``, ``"hard_8agent"``.
    num_agents : int, optional
        Override the number of agents (default: read from task_id).
    seed : int
        Global random seed.
    agent_fn : callable, optional
        ``(agent_id, observation) -> action_id``.
        Defaults to the built-in rule-based agent.
    verbose : bool
        Print step messages when True.

    Returns
    -------
    MultiAgentResult
        Episode scoring result.
    """
    from .models import MultiAgentAction

    env = create_env_for_task(task_id)
    env.reset(seed=seed)
    n = env._ma_cfg.num_agents
    trajectories: dict = {i: [] for i in range(n)}

    fn = agent_fn or (lambda aid, obs: _rule_based_action(obs))
    done_agents: set = set()
    max_steps = env._env_cfg.max_steps * n
    total_steps = 0

    while len(done_agents) < n and total_steps < max_steps:
        for agent_id in env.get_turn_order():
            if agent_id in done_agents:
                continue
            obs = env.get_obs(agent_id)
            if obs.done:
                done_agents.add(agent_id)
                continue
            action_id = fn(agent_id, obs)
            action = MultiAgentAction(action_id=action_id, agent_id=agent_id)
            new_obs = env.step(action)
            trajectories[agent_id].append({
                "prices": [
                    new_obs.market_price_crop_1,
                    new_obs.market_price_crop_2,
                    new_obs.market_price_crop_3,
                ]
            })
            if verbose:
                print(f"[A{agent_id} s{new_obs.current_step}] {new_obs.message[:80]}")
            if new_obs.done:
                done_agents.add(agent_id)
        total_steps += 1

    return env.compute_result(trajectories)

