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
from typing import List, Optional

from openai import OpenAI

from cropRL.tasks import create_env_for_task, grader
from cropRL.models import CroprlAction

# ── Configuration ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma4:e4b")
TEMPERATURE = 0.0  # Set to 0 to prevent erratic thinking tokens
MAX_TOKENS = 10  # Hugely reduced to prevent the model from rambling or thinking

SYSTEM_PROMPT = """\
You are an expert farm manager AI. You manage a small Indian farm over 60 months.

OBJECTIVE: Maximize your net worth (cash + crop value - debt + soil bonus) by the end of 60 months.

ACTIONS (reply with ONLY the action number):
0: Wait — Do nothing this month.
1: Plant Corn — High cost (₹800), high yield, depletes soil nitrogen heavily.
2: Plant Wheat — Moderate cost (₹500), moderate yield, mild nitrogen drain.
3: Plant Chickpea — Low cost (₹200), lower yield, RESTORES soil nitrogen.
4: Irrigate — Costs ₹300, mitigates drought impact on growing crops.
5: Fertilize — Costs ₹400, boosts soil nitrogen by 0.15.
6: Harvest & Store — Harvest crop and store it (auto-sells old storage).
7: Harvest & Sell — Harvest crop and sell immediately at market price.
8: Sell Inventory — Sell stored crops at current market price.
9: Take Loan — Get ₹5,000 (only if no active loan). Interest accumulates monthly.
10: Repay Loan — Pay off full debt (must have enough cash).

KEY RULES:
- Can only plant on fallow (empty) land.
- Can only harvest crops aged >= 1 month. Crops mature at 3-4 months for full yield.
- Storage rots after 6 months. Only one slot.
- One loan at a time. Must repay full amount.
- Soil nitrogen is crucial: low N = poor yields. Chickpeas restore N, Corn destroys it.
- Bankruptcy (negative cash + loan) ends the game with heavy penalty.
- At month 60, remaining soil health and crop/storage value give terminal bonus.

CRITICAL INSTRUCTION:
DO NOT use <think> tags.
DO NOT output any reasoning, chain-of-thought, or explanation.
Respond IMMEDIATELY with ONLY a single integer (0-10).
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
    Action IDs:
    0=Wait, 1=Corn, 2=Wheat, 3=Chickpea, 4=Irrigate, 5=Fertilize,
    6=Harvest & Store, 7=Harvest & Sell, 8=Sell Inventory, 9=Take Loan, 10=Repay Loan
    """
    # 1. Clear inventory first if any
    if obs.stored_amount > 0:
        return 8  # Sell Inventory
        
    # 2. Plant if land is fallow
    if obs.active_crop_type == 0:
        # If soil nitrogen is low, plant restorative crop (Chickpea)
        if obs.soil_nitrogen < 0.4 and obs.cash_balance >= obs.cost_seed_3:
            return 3
        # If we have lots of cash and decent soil, plant high-yield (Corn)
        elif obs.cash_balance >= obs.cost_seed_1 and obs.soil_nitrogen > 0.5:
            return 1
        # Otherwise plant moderate (Wheat)
        elif obs.cash_balance >= obs.cost_seed_2:
            return 2
        # Failsafe if broke but no loan
        elif obs.cash_balance < obs.cost_seed_3 and not getattr(obs, "has_active_loan", False):
            # We don't have enough to plant the cheapest crop. Take a loan if we don't have one!
            # Wait, has_active_loan might not be in Observation, it's in State. 
            # We can guess by looking at debt.
            if obs.current_debt == 0:
                return 9 # Take Loan
        return 0  # Wait
        
    # 3. Manage growing crop
    if obs.active_crop_type > 0:
        # If crop is mature enough
        if obs.crop_age_months >= 4:
            return 7  # Harvest & Sell
        elif obs.crop_age_months >= 3 and obs.expected_yield_potential > 0.8:
            return 7  # Harvest & Sell early if yield is good
            
        # Optional: Fertilize if soil is very low
        if obs.soil_nitrogen < 0.2 and obs.cash_balance >= obs.cost_fertilize:
            return 5
            
        # Optional: Irrigate if expected rainfall is very low
        if obs.expected_rainfall < 0.3 and obs.cash_balance >= obs.cost_irrigate:
            return 4
            
    return 0  # Wait


def parse_action(response_text: str, fallback_action: int) -> int:
    """Extract an action integer from the LLM response."""
    cleaned = response_text.strip()
    if cleaned.isdigit():
        val = int(cleaned)
        if 0 <= val <= 10:
            return val
    matches = re.findall(r"\b(\d{1,2})\b", cleaned)
    for match in matches:
        val = int(match)
        if 0 <= val <= 10:
            return val
    return fallback_action


def get_model_action(client: OpenAI, obs, history: List[str]) -> int:
    fallback = rule_based_agent(obs)
    user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
    
    history_block = "\n".join(history[-12:]) if history else "None"
    user_msg += f"\n\nRecent History:\n{history_block}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response = completion.choices[0].message.content or ""
        return parse_action(response, fallback)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr)
        return fallback


def run_episode(client: OpenAI, task_id: str):
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
                
            action_id = get_model_action(client, obs, history)
            action_name = env.config.action_names[action_id]
            
            action = CroprlAction(action_id=action_id)
            obs = env.step(action)
            
            reward = obs.reward or 0.0
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_name, reward=reward, done=done, error=None)
            
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
            
            if done:
                break
                
        # Calculate score using the grader
        final_net_worth = (
            obs.cash_balance - obs.current_debt + obs.soil_nitrogen * 10000
        )
        score = grader(task_id, final_net_worth, obs.done and steps_taken < env.config.max_steps, trajectory)
        
        # Consider successful if score >= 0.1
        success = score >= 0.1
        
    except Exception as e:
        print(f"[DEBUG] Error during episode execution: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    for task in ["easy", "medium", "hard"]:
        run_episode(client, task)


if __name__ == "__main__":
    main()