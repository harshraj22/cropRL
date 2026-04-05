"""
CropRL REST API — FastAPI interface to the farming environment.

Exposes the environment via HTTP so any client (curl, browser, JS, Python)
can reset, query state, and take actions.

Usage:
    uvicorn extras.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException

# ── Logging setup ──────────────────────────────────────────────
logger = logging.getLogger("croprl.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.server.cropRL_environment import CroprlEnvironment
from cropRL.tasks import TASKS, create_env_for_task

from .schemas import (
    ActionRequest,
    ActionResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    TaskInfo,
)

app = FastAPI(
    title="CropRL API",
    description="REST API for the CropRL farming reinforcement learning environment.",
    version="0.1.0",
)

# ── Global environment instance ────────────────────────────────
_env: CroprlEnvironment | None = None
_last_obs = None
_episode_step = 0


def _get_env() -> CroprlEnvironment:
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active environment. Call POST /reset first.",
        )
    return _env


# ── Endpoints ──────────────────────────────────────────────────


@app.get("/", tags=["info"])
def root():
    """Health check."""
    return {"status": "ok", "service": "croprl-api"}


@app.get("/tasks", response_model=list[TaskInfo], tags=["info"])
def list_tasks():
    """List all available task difficulties."""
    result = []
    for task_id, task_def in TASKS.items():
        result.append(
            TaskInfo(
                task_id=task_id,
                description=task_def["description"],
                max_steps=task_def.get("config_overrides", {}).get("max_steps", 60),
            )
        )
    return result


@app.get("/actions", tags=["info"])
def list_actions():
    """List all possible actions with their IDs and names."""
    cfg = EnvConfig()
    return {
        "actions": [
            {"id": i, "name": name}
            for i, name in enumerate(cfg.action_names)
        ]
    }


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
def reset_env(req: ResetRequest | None = None):
    """Reset the environment and start a new episode.

    Optionally specify a task difficulty and random seed.
    """
    global _env, _last_obs, _episode_step

    task_id = (req.task if req and req.task else "medium")
    seed = (req.seed if req and req.seed is not None else 42)
    text_mode = (req.text_mode if req else True)

    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_id}'. Available: {list(TASKS.keys())}",
        )

    _env = create_env_for_task(task_id, text_mode=text_mode)
    _last_obs = _env.reset(seed=seed)
    _episode_step = 0

    logger.info(
        f"RESET  │ task={task_id:<6s} seed={seed:<4d} │"
        f" cash=₹{_last_obs.cash_balance:,.0f}"
        f"  soil={_last_obs.soil_nitrogen:.2f}"
        f"  steps={_env.config.max_steps}"
    )

    return ResetResponse(
        message=f"Environment reset. Task: {task_id}, Seed: {seed}",
        task_id=task_id,
        observation=_obs_to_dict(_last_obs),
    )


@app.get("/state", response_model=StateResponse, tags=["environment"])
def get_state():
    """Get the current environment observation."""
    env = _get_env()
    if _last_obs is None:
        raise HTTPException(status_code=400, detail="No observation yet. Call POST /reset first.")

    return StateResponse(
        step=_episode_step,
        done=_last_obs.done,
        observation=_obs_to_dict(_last_obs),
    )


@app.post("/step", response_model=ActionResponse, tags=["environment"])
def take_action(req: ActionRequest):
    """Take an action in the environment.

    Pass an action_id (0-10). Returns the resulting observation and reward.
    """
    global _last_obs, _episode_step

    env = _get_env()

    if _last_obs and _last_obs.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new one.",
        )

    if not (0 <= req.action_id <= 10):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_id {req.action_id}. Must be 0-10.",
        )

    action = CroprlAction(action_id=req.action_id)
    _last_obs = env.step(action)
    _episode_step += 1

    cfg = env.config
    action_name = cfg.action_names[req.action_id]

    # Log step details
    crop_name = cfg.crop_names[_last_obs.active_crop_type]
    crop_info = (
        f"{crop_name}({_last_obs.crop_age_months}m)"
        if _last_obs.active_crop_type > 0
        else "--"
    )
    reward_val = _last_obs.reward or 0
    done_tag = "  [DONE]" if _last_obs.done else ""
    logger.info(
        f"STEP {_episode_step:2d} │ {action_name:<26s} │"
        f" cash=₹{_last_obs.cash_balance:>9,.0f}"
        f"  debt=₹{_last_obs.current_debt:,.0f}"
        f"  soil={_last_obs.soil_nitrogen:.2f}"
        f"  crop={crop_info:<14s} │"
        f" R={reward_val:+.0f}{done_tag}"
    )

    return ActionResponse(
        step=_episode_step,
        action_id=req.action_id,
        action_name=action_name,
        reward=_last_obs.reward,
        done=_last_obs.done,
        message=_last_obs.message,
        observation=_obs_to_dict(_last_obs),
    )


# ── Helpers ────────────────────────────────────────────────────


def _obs_to_dict(obs) -> dict:
    """Convert a CroprlObservation to a JSON-friendly dict."""
    return {
        "current_month": obs.current_month,
        "current_step": obs.current_step,
        "expected_rainfall": round(obs.expected_rainfall, 4),
        "active_crop_type": obs.active_crop_type,
        "crop_age_months": obs.crop_age_months,
        "expected_yield_potential": round(obs.expected_yield_potential, 4),
        "soil_nitrogen": round(obs.soil_nitrogen, 4),
        "current_water_level": round(obs.current_water_level, 4),
        "cash_balance": round(obs.cash_balance, 2),
        "current_debt": round(obs.current_debt, 2),
        "current_interest_rate": round(obs.current_interest_rate, 4),
        "current_land_price": round(obs.current_land_price, 2),
        "market_prices": {
            "corn": round(obs.market_price_crop_1, 2),
            "wheat": round(obs.market_price_crop_2, 2),
            "chickpea": round(obs.market_price_crop_3, 2),
        },
        "costs": {
            "seed_corn": obs.cost_seed_1,
            "seed_wheat": obs.cost_seed_2,
            "seed_chickpea": obs.cost_seed_3,
            "irrigate": obs.cost_irrigate,
            "fertilize": obs.cost_fertilize,
        },
        "storage": {
            "crop_type": obs.stored_crop_type,
            "amount": round(obs.stored_amount, 2),
            "age_months": obs.stored_age_months,
        },
        "message": obs.message,
        "text_summary": obs.text_summary if obs.text_summary else None,
        "done": obs.done,
    }
