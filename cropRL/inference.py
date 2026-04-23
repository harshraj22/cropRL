"""
Inference Script for CropRL Environment
=================================================

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Any, List, Optional, Dict

# Ensure the root directory is on the path so cropRL module works anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import ollama

from cropRL.tasks import create_env_for_task, grader, TASKS
from cropRL.models import MultiAgentAction
from cropRL.enums import ActionType, CropType

# ── Configuration ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5:9b")
TEMPERATURE = 0.0  # Set to 0 to prevent erratic thinking tokens

SYSTEM_PROMPT = """\
You are an expert farm manager AI. You manage a small Indian farm over 60 months.
You may be competing or cooperating with other AI farmers in the village.

OBJECTIVE: Maximize your net worth (cash + land value + crop value - debt) by the end of 60 months.

ACTIONS:
0: Wait / No-Op — Do nothing but consume 1 action slot.
1: Plant Corn — High cost, high yield, depletes soil nitrogen heavily. 
2: Plant Wheat — Moderate cost/yield, mild nitrogen drain. Best in Winter.
3: Plant Chickpea — Low cost, lower yield, RESTORES soil nitrogen. 
4: Irrigate — Adds water to field instantly. Critical during dry months.
5: Fertilize — Boosts soil nitrogen by 0.15 instantly.
6: Harvest & Store — Harvest crop and store it (auto-sells old storage).
7: Harvest & Sell — Harvest crop and queue sale for month-end clearing.
8: Sell Inventory — Queue stored crops for month-end sale.
9: Take Loan — Get cash (only if no active loan). Interest locked at current rate.
10: Repay Loan — Pay off full debt (must have enough cash).
11: Post Forum Message — Send a short intent message to other agents.
12: Plant Matcha (Hype Crop) — High hype premium but saturates fast.
13: Plant Quinoa (Hype Crop) — Moderate hype premium.
14: Plant Turmeric (Hype Crop) — Moderate hype premium.

KEY RULES:
- Action 0 (Wait) consumes an action slot and does nothing else. The month advances ONLY when all agents expend all configured action slots.
- Actions cost 1 action slot each month.
- Crops queued to sell are cleared at the END of the month. High supply drops the market clearing price for everyone.
- Hype crops follow unpredictable cycles. Monitor Social Media Trends.
- Can only plant on fallow (empty) land.
- Can only harvest crops aged >= 1 month. 
- Storage rots after 6 months. Only one slot.
- One loan at a time. Must repay full amount. Interest uses rate when loan was taken.
- Soil nitrogen is crucial: low N = poor yields. Chickpeas restore N, Corn destroys it.
- Water level matters.
- Growing crops in their optimal season gives much better yields.
- Inflation increases costs each year.
- Monthly fixed costs are deducted every month.
- Bankruptcy (negative cash + loan) ends the game with heavy penalty.

CRITICAL INSTRUCTION:
You MUST respond with a valid JSON object. Do not provide any conversational text before or after the JSON.
The JSON object must have exactly the following fields:
- "thought": (string) Your reasoning for choosing the action based on the current state.
- "action": (integer) The action number you choose (0-14).
- "message": (string) An optional message. If you chose action 11, this must contain your message to other agents. Otherwise, you can leave it empty or null.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def rule_based_agent(obs) -> int:
    """
    Deterministic rule-based agent for CropRL environment.
    """
    # 1. Clear inventory first if any
    if obs.stored_amount > 0:
        return ActionType.SELL_INVENTORY

    # 2. Plant if land is fallow
    if obs.active_crop_type == CropType.FALLOW:
        # If soil nitrogen is low, plant restorative crop (Chickpea)
        if obs.soil_nitrogen < 0.4 and obs.cash_balance >= getattr(obs, "cost_seed_3", 200.0):
            return ActionType.PLANT_CHICKPEA
        # If we have lots of cash and decent soil, maybe plant Hype or Corn
        elif obs.cash_balance >= 1500 and obs.soil_nitrogen > 0.5:
            # Just default to corn, hype is risky for rules
            return ActionType.PLANT_CORN
        elif obs.cash_balance >= getattr(obs, "cost_seed_1", 800.0) and obs.soil_nitrogen > 0.5:
            return ActionType.PLANT_CORN
        # Otherwise plant moderate (Wheat)
        elif obs.cash_balance >= getattr(obs, "cost_seed_2", 500.0):
            return ActionType.PLANT_WHEAT
        # Failsafe if broke
        elif obs.cash_balance < getattr(obs, "cost_seed_3", 200.0) and obs.current_debt == 0:
            return ActionType.TAKE_LOAN
        return ActionType.WAIT

    # 3. Manage growing crop
    if obs.active_crop_type != CropType.FALLOW:
        # If crop is mature enough, harvest & sell
        if obs.crop_age_months >= 4:
            return ActionType.HARVEST_SELL
        elif obs.crop_age_months >= 3 and obs.expected_yield_potential > 0.8:
            return ActionType.HARVEST_SELL

        # Fertilize if soil is very low
        if obs.soil_nitrogen < 0.2 and obs.cash_balance >= getattr(obs, "cost_fertilize", 300.0):
            return ActionType.FERTILIZE

        # Irrigate if water is low
        if obs.current_water_level < 0.2 and obs.cash_balance >= getattr(obs, "cost_irrigate", 300.0):
            return ActionType.IRRIGATE

    return ActionType.WAIT


def parse_action(response_text: str, fallback_action: int) -> tuple[int, Optional[str]]:
    """Extract an action integer and optional message from the LLM response."""
    try:
        data = json.loads(response_text)
        action = int(data.get("action", fallback_action))
        message = data.get("message")
        return action, message
    except Exception as e:
        print(f"[DEBUG] Failed to parse JSON response: {e}")
        return fallback_action, None


def get_agent_system_prompt(agent_id: int, num_agents: int) -> str:
    """Build a per-agent system prompt with identity context."""
    return SYSTEM_PROMPT + (
        f"\n\nAGENT IDENTITY:\n"
        f"You are Agent {agent_id} (out of {num_agents} farmers in this village).\n"
        f"Your farm is independent — you have your own land, cash, and crops.\n"
        f"You can see what other agents plant (via the observation) and \n"
        f"communicate via the Forum. Coordinate to avoid saturating the market \n"
        f"with the same crop — if multiple agents sell the same crop, the \n"
        f"clearing price drops for everyone.\n"
    )


def get_model_action(
    client: Any, obs, history: List[str],
    agent_id: Optional[int] = None, num_agents: int = 1,
) -> tuple[int, Optional[str]]:
    fallback = rule_based_agent(obs)
    user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)

    history_block = "\n".join(history[-12:]) if history else "None"
    user_msg += f"\n\nRecent History:\n{history_block}"

    # Use per-agent prompt if agent_id is provided (multi-agent mode)
    if agent_id is not None:
        prompt = get_agent_system_prompt(agent_id, num_agents)
    else:
        prompt = SYSTEM_PROMPT

    print(f"\n[LLM PROMPT] Agent {agent_id if agent_id is not None else 0}\nSYSTEM:\n{prompt}\nUSER:\n{user_msg}\n", flush=True)

    try:
        response_obj = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            options={"temperature": TEMPERATURE}
        )
        response_text = response_obj['message']['content']
        print(f"\n[LLM RESPONSE] Agent {agent_id if agent_id is not None else 0}\n{response_text}\n", flush=True)
        return parse_action(response_text, fallback)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr)
        return fallback, None


def run_single_agent_episode(client: Any, task_id: str):
    """Run a single-agent episode using MultiAgentCroprlEnvironment with num_agents=1."""
    env = create_env_for_task(task_id, text_mode=True)
    env.reset(seed=42)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="croprl", model=MODEL_NAME)
    max_steps = env._env_cfg.max_steps
    trajectory: list = []

    try:
        for step in range(1, max_steps + 1):
            # Always fetch fresh observation
            obs = env.get_obs(0)

            if obs.done:
                break

            obs_details = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
            print(f"\n[OBSERVATION - Step {step}]\n{obs_details}\n", flush=True)

            action_id, forum_message = get_model_action(client, obs, history, agent_id=0, num_agents=1)
            action_name = env._env_cfg.action_names[action_id] if action_id < len(env._env_cfg.action_names) else f"Action {action_id}"

            action = MultiAgentAction(action_id=action_id, agent_id=0, forum_message=forum_message)
            result_obs = env.step(action)

            reward = result_obs.reward or 0.0
            done = result_obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_name, reward=reward, done=done, error=None)

            history.append(f"Step {step}: Selected '{action_name}' -> Reward {reward:+.2f}")

            trajectory.append({
                "step": step,
                "action_id": action_id,
                "reward": reward,
                "cash": result_obs.cash_balance,
                "debt": result_obs.current_debt,
                "soil_n": result_obs.soil_nitrogen,
                "prices": [
                    result_obs.market_price_crop_1,
                    result_obs.market_price_crop_2,
                    result_obs.market_price_crop_3,
                    result_obs.market_price_crop_4,
                    result_obs.market_price_crop_5,
                    result_obs.market_price_crop_6,
                ]
            })

            if done:
                break

        # Use compute_result for consistent scoring
        result = env.compute_result({0: trajectory})
        score = result.aggregate_score
        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Error during episode execution: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_multi_agent_episode_llm(client: Any, task_id: str):
    """Run a multi-agent episode with LLM agents."""
    env = create_env_for_task(task_id, text_mode=True)
    env.reset(seed=42)
    n = env._ma_cfg.num_agents

    histories: Dict[int, List[str]] = {i: [] for i in range(n)}
    trajectories: Dict[int, List[dict]] = {i: [] for i in range(n)}
    done_agents: set = set()

    max_steps = env._env_cfg.max_steps * n
    total_steps = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="croprl_multi_agent", model=MODEL_NAME)

    try:
        while len(done_agents) < n and total_steps < max_steps:
            for agent_id in env.get_turn_order():
                # Always fetch fresh observation — no caching needed
                obs = env.get_obs(agent_id)

                if obs.done:
                    done_agents.add(agent_id)
                    # Dead/done agents automatically wait out their slots so they don't block TimeController
                    action_id = 0
                    forum_message = None
                else:
                    action_id, forum_message = get_model_action(client, obs, histories[agent_id], agent_id=agent_id, num_agents=n)
                
                action_name = env._env_cfg.action_names[action_id] if action_id < len(env._env_cfg.action_names) else f"Action {action_id}"

                action = MultiAgentAction(action_id=action_id, agent_id=agent_id, forum_message=forum_message)
                new_obs = env.step(action)

                reward = new_obs.reward or 0.0
                total_steps += 1

                log_step(step=total_steps, action=f"A{agent_id}:{action_name}", reward=reward, done=new_obs.done, error=None)

                histories[agent_id].append(f"Step {new_obs.current_step}: Selected '{action_name}' -> Reward {reward:+.2f}")

                # Trajectory bookkeeping
                trajectories[agent_id].append({
                    "step": new_obs.current_step,
                    "action_id": action_id,
                    "reward": reward,
                    "cash": new_obs.cash_balance,
                    "debt": new_obs.current_debt,
                    "soil_n": new_obs.soil_nitrogen,
                    "prices": [
                        new_obs.market_price_crop_1,
                        new_obs.market_price_crop_2,
                        new_obs.market_price_crop_3,
                        new_obs.market_price_crop_4,
                        new_obs.market_price_crop_5,
                        new_obs.market_price_crop_6,
                    ]
                })

                # Only print observation detail if they actually took a choice (aren't dead yet)
                if not obs.done:
                    obs_details = new_obs.text_summary if getattr(new_obs, "text_summary", None) else str(new_obs)
                    print(f"\n[OBSERVATION - A{agent_id} Step {new_obs.current_step}]\n{obs_details}\n", flush=True)

                if new_obs.done:
                    done_agents.add(agent_id)

        result = env.compute_result(trajectories)
        score = result.aggregate_score
        success = score >= 0.1
        log_end(success=success, steps=total_steps, score=score, rewards=list(result.agent_scores.values()))

    except Exception as e:
        print(f"[DEBUG] Error during multi-agent episode execution: {e}", flush=True)
        log_end(success=False, steps=total_steps, score=0.0, rewards=[])


def run_episode(client: Any, task_id: str):
    task_info = TASKS.get(task_id, {})
    if task_info.get("multi_agent", False):
        run_multi_agent_episode_llm(client, task_id)
    else:
        run_single_agent_episode(client, task_id)


def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser(description="Run CropRL inference")
    parser.add_argument("--task", type=str, default="easy", help="Task ID to run")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    args = parser.parse_args()
    
    MODEL_NAME = args.model

    client = ollama.Client(host=API_BASE_URL)
    # Run task
    run_episode(client, args.task)


if __name__ == "__main__":
    main()