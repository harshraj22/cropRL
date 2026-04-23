"""CropRL Task Definitions."""
from __future__ import annotations
from .config import EnvConfig
from .multi_env import MultiAgentCroprlEnv

TASKS: dict[str, dict] = {
    "easy": {"description": "Maximize net worth over 60 months. Simplified conditions.",
             "config_overrides": {"initial_cash": 15000.0, "base_interest_rate": 0.0,
              "weather_sigma": 0.05, "weather_sigma_realisation": 0.02,
              "market_price_sigma": 0.05, "initial_soil_nitrogen": 0.8,
              "max_storage_age": 12, "monthly_fixed_cost": 100.0}},
    "medium": {"description": "Maximize net worth over 60 months. Standard conditions.",
               "config_overrides": {}},
    "hard": {"description": "Maximize net worth over 60 months. Harsh conditions.",
             "config_overrides": {"initial_cash": 7000.0, "base_interest_rate": 0.12,
              "weather_sigma": 0.25, "weather_sigma_realisation": 0.08,
              "market_price_sigma": 0.20, "initial_soil_nitrogen": 0.35,
              "max_storage_age": 4, "monthly_fixed_cost": 300.0}},
}

def create_env_for_task(task_id: str, text_mode: bool = False, num_farmers: int = 2) -> MultiAgentCroprlEnv:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}")
    overrides = TASKS[task_id]["config_overrides"].copy()
    overrides["text_mode"] = text_mode
    overrides["num_farmers"] = num_farmers
    return MultiAgentCroprlEnv(config=EnvConfig(**overrides), task_id=task_id)

def list_tasks() -> dict[str, str]:
    return {tid: info["description"] for tid, info in TASKS.items()}

def grader(task_id: str, final_net_worths, bankrupt: bool, trajectory=None) -> float:
    avg_nw = sum(final_net_worths.values()) / max(1, len(final_net_worths)) if isinstance(final_net_worths, dict) else final_net_worths
    if bankrupt or avg_nw <= 0 or not trajectory: return 0.01
    cfg = EnvConfig(**TASKS.get(task_id, {}).get("config_overrides", {}))
    baseline = cfg.initial_cash - (cfg.max_months * cfg.monthly_fixed_cost) + cfg.base_land_price * cfg.initial_soil_nitrogen
    if avg_nw <= baseline: return 0.01
    total_oracle = sum(max(0.0, ((s.get("prices", [1200,800,500])[0]*8-800)/4),
                          ((s.get("prices", [1200,800,500])[1]*5-500)/3),
                          ((s.get("prices", [1200,800,500])[2]*3-200)/3)) for s in trajectory)
    oracle_max = cfg.initial_cash + cfg.base_land_price + total_oracle
    if oracle_max <= baseline: return 0.5
    return float(max(0.01, min(0.99, (avg_nw - baseline) / (oracle_max - baseline))))

class EasyGrader:
    def grade(self, nw, bankrupt, traj=None): return max(0.01, min(0.99, grader("easy", nw, bankrupt, traj)))
class MediumGrader:
    def grade(self, nw, bankrupt, traj=None): return max(0.01, min(0.99, grader("medium", nw, bankrupt, traj)))
class HardGrader:
    def grade(self, nw, bankrupt, traj=None): return max(0.01, min(0.99, grader("hard", nw, bankrupt, traj)))
