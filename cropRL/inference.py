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
import time
from pathlib import Path
from typing import List, Optional

# Ensure the root directory is on the path so cropRL module works anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from openai import OpenAI

from cropRL.tasks import create_env_for_task, grader
from cropRL.models import CroprlAction
from cropRL.enums import ActionType, CropType

# ── Configuration ──────────────────────────────────────────────
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "ollama")

if USE_OPENROUTER:
    # MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.6-plus:free")
    MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31b-it:free")
else:
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:8b")

# Multi-model fallback list for OpenRouter free tier
OPENROUTER_FALLBACK_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free"
]

# Proactive Rotation State
CURRENT_MODEL_INDEX = 0

# Default temperature (thinking level)
TEMPERATURE = 0.4  # Creative reasoning

# Medium thinking level for Ollama qwen3.5:9b model
if not USE_OPENROUTER and MODEL_NAME.startswith("qwen3.5:9b"):
    TEMPERATURE = 0.6  # Medium thinking

MAX_TOKENS = 8192
STEP_DELAY = float(os.getenv("STEP_DELAY", "2.0"))  # Seconds to sleep between steps to avoid rate limits
RETRY_BASE_DELAY = 2.0  # Base delay for exponential backoff during rate limits

SYSTEM_PROMPT = """\
You are an expert farm manager AI. You manage a small Indian farm over 60 months.

OBJECTIVE: Maximize your net worth (cash + land value + crop value - debt) by the end of 60 months.

ACTIONS (use the action integer):
0: Wait — End this month and advance to the next. Monthly costs deducted.
1: Plant Corn
2: Plant Wheat
3: Plant Chickpea
4: Irrigate — Adds water to field instantly. Different crops have different amount of water added to them by irrigation.
5: Fertilize — Boosts soil nitrogen by 0.15 instantly.
6: Harvest & Store — Harvest crop and store it (auto-sells old storage).
7: Harvest & Sell — Harvest crop and sell immediately at market price.
8: Sell Inventory — Sell stored crops at current market price.
9: Take Loan — Get cash (only if no active loan). Interest locked at current rate.
10: Repay Loan — Pay off full debt (must have enough cash).
NOTE: The water_level metric in observations is normalized to a maximum of 1, representing the land's full water holding capacity. Different crops have different optimal water requirements.
THINKING: Keep your chain-of-thought concise and reasonable, avoiding overly long deliberations.

KEY RULES:
- Only Wait (action 0) advances the calendar month. Other actions are instant.
- Can only plant on fallow (empty) land.
- Can only harvest crops aged >= 1 month.
- Only one crop can be stored at a time.
- One loan at a time. Must repay full amount. Interest uses rate when loan was taken.
- Monthly fixed costs are deducted every month.
- Bankruptcy (negative cash + loan) ends the game with heavy penalty.

OUTPUT FORMAT:
Your response MUST be a valid JSON object with the following fields:
1. "thought": A detailed chain-of-thought analysis of the current situation (season, finances, soil, weather) and your strategy.
2. "action_id": The integer ID (0-10) of your chosen action.

Example:
{
  "thought": "It is July (Monsoon), soil nitrogen is high, and I have enough cash. Planting corn is optimal now.",
  "action_id": 1
}
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
        if obs.soil_nitrogen < 0.4 and obs.cash_balance >= obs.cost_seed_3:
            return ActionType.PLANT_CHICKPEA
        # If we have lots of cash and decent soil, plant high-yield (Corn)
        elif obs.cash_balance >= obs.cost_seed_1 and obs.soil_nitrogen > 0.5:
            return ActionType.PLANT_CORN
        # Otherwise plant moderate (Wheat)
        elif obs.cash_balance >= obs.cost_seed_2:
            return ActionType.PLANT_WHEAT
        # Failsafe if broke
        elif obs.cash_balance < obs.cost_seed_3 and obs.current_debt == 0:
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
        if obs.soil_nitrogen < 0.2 and obs.cash_balance >= obs.cost_fertilize:
            return ActionType.FERTILIZE

        # Irrigate if water is low
        if obs.current_water_level < 0.2 and obs.cash_balance >= obs.cost_irrigate:
            return ActionType.IRRIGATE

    return ActionType.WAIT


import json

def parse_action(response_text: str, fallback_action: int) -> int:
    """Extract an action integer from the LLM response, prioritizing JSON."""
    cleaned = response_text.strip()

    # 1. Try strict JSON
    try:
        data = json.loads(cleaned)
        if "action_id" in data:
            val = int(data["action_id"])
            if 0 <= val <= 10:
                return val
    except:
        pass

    # 2. Try to find JSON block in text
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "action_id" in data:
                val = int(data["action_id"])
                if 0 <= val <= 10:
                    return val
        except:
            pass

    # 3. Fallback to original digit extraction
    if cleaned.isdigit():
        val = int(cleaned)
        if 0 <= val <= 10:
            return val
    matches = re.findall(r"\b(\d{1,2})\b", cleaned)
    for match in matches:
        val = int(match)
        if 0 <= val <= 10:
            return val
    print("Using fallback action: ", fallback_action)
    return fallback_action


def get_model_action(obs, history: List[str]) -> int:
    global CURRENT_MODEL_INDEX
    fallback = rule_based_agent(obs)
    user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)

    # Use larger history window for high-potential models
    history_block = "\n".join(history[-50:]) if history else "None"
    user_msg += f"\n\nRecent History:\n{history_block}"

    # Debug: Print the exact message being sent to the LLM
    print(f"\n--- [LLM PROMPT] ---\n{user_msg}\n--- [/LLM PROMPT] ---\n", flush=True)

    base_models = [MODEL_NAME]
    if USE_OPENROUTER:
        base_models += OPENROUTER_FALLBACK_MODELS

    # Reorder models based on CURRENT_MODEL_INDEX for rotation
    models_count = len(base_models)
    models_to_try = [base_models[(CURRENT_MODEL_INDEX + i) % models_count] for i in range(models_count)]

    for attempt, current_model in enumerate(models_to_try):
        try:
            response_text = ""
            print(f"\n--- Model Reasoning ({current_model} | Step {obs.current_step}) ---", flush=True)

            if USE_OPENROUTER:
                client = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=OPENROUTER_API_KEY,
                    default_headers={
                        "HTTP-Referer": "https://github.com/OpenEnv/CropRL",
                        "X-Title": "CropRL Inference",
                    }
                )

                stream = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        response_text += content
            else:
                stream = ollama.chat(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    stream=True,
                    options={
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                    }
                )

                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    response_text += content

            print(f"\n--- End Reasoning ---\n", flush=True)

            # Successfully got action, update index for next call to rotate
            internal_idx = base_models.index(current_model)
            CURRENT_MODEL_INDEX = (internal_idx + 1) % models_count

            return parse_action(response_text, fallback)

        except Exception as e:
            if attempt < models_count - 1:
                wait_time = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"\n[ERROR] Error on {current_model}: {e}. Switching models in {wait_time:.1f}s...", file=sys.stderr)
                time.sleep(wait_time)
                continue # Pivot to next model in the list
            else:
                provider = "OpenRouter" if USE_OPENROUTER else "Ollama"
                print(f"[DEBUG] {provider} final failure on {current_model}: {e}", file=sys.stderr)
                # Keep current index for next step attempt
                return fallback

    return fallback


def run_episode(task_id: str):
    # Pass text_mode=True so obs has a .text_summary
    env = create_env_for_task(task_id, text_mode=True)
    obs = env.reset(seed=42)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="croprl", model=MODEL_NAME)

    trajectory = []

    try:
        for step in range(1, env.config.max_steps + 1):
            if obs.done:
                break

            action_id = get_model_action(obs, history)
            action_name = env.config.action_names[action_id]

            action = CroprlAction(action_id=action_id)
            obs = env.step(action)

            reward = obs.reward or 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_name, reward=reward, done=done, error=None)

            # Formatted Observation Logging
            obs_details = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
            print(f"\n[OBSERVATION - Step {step}]\n{obs_details}\n", flush=True)

            history.append(f"Step {step}: Selected '{action_name}' -> Reward {reward:+.2f}")

            trajectory.append({
                "step": step,
                "action_id": action_id,
                "reward": reward,
                "cash": obs.cash_balance,
                "debt": obs.current_debt,
                "soil_n": obs.soil_nitrogen,
                "prices": [
                    obs.market_price_crop_1,
                    obs.market_price_crop_2,
                    obs.market_price_crop_3,
                ]
            })

            # Rate limiting protection
            if STEP_DELAY > 0:
                time.sleep(STEP_DELAY)

            if done:
                break

        # Calculate score using the grader
        final_net_worth = (
            obs.cash_balance
            - obs.current_debt
            + obs.current_land_price
            + (obs.stored_amount * obs.market_price_crop_1  # approximate
               if obs.stored_crop_type > 0 else 0)
        )
        score = grader(
            task_id, final_net_worth,
            obs.done and obs.cash_balance < 0,
            trajectory,
        )

        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Error during episode execution: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    for task in ["medium"]: #["easy", "medium", "hard"]:
        run_episode(task)


if __name__ == "__main__":
    main()