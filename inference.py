"""
Rule-Based Inference Script for CropRL Environment
=================================================

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import os
from typing import List, Optional

from crop_env.tasks import create_env_for_task, grader
from crop_env.models import CropAction

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


def run_episode(task_id: str):
    env = create_env_for_task(task_id, text_mode=False)
    obs = env.reset(seed=42)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_id, env="croprl", model="rule-based")
    
    trajectory = []
    
    try:
        for step in range(1, env.config.max_steps + 1):
            if obs.done:
                break
                
            action_id = rule_based_agent(obs)
            action_name = env.config.action_names[action_id]
            
            action = CropAction(action_id=action_id)
            obs = env.step(action)
            
            reward = obs.reward or 0.0
            done = obs.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_name, reward=reward, done=done, error=None)
            
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


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_episode(task)