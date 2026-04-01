"""
Inference Script for CropRL Environment.
===================================
MANDATORY:
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

Usage:
    API_BASE_URL="https://..." MODEL_NAME="meta-llama/..." HF_TOKEN="..." python inference.py
"""

import json
import os
import re
import sys

from openai import OpenAI

from crop_env.config import EnvConfig
from crop_env.models import CropAction
from crop_env.server.crop_environment import CropEnvironment
from crop_env.tasks import TASKS, create_env_for_task, grader

# ── Configuration ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
MAX_STEPS = 60
TEMPERATURE = 0.2
MAX_TOKENS = 150
FALLBACK_ACTION = 0  # Wait

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

Respond with ONLY a single integer (0-10). No explanation needed.
"""


def parse_action(response_text: str) -> int:
    """Extract an action integer from the LLM response."""
    cleaned = response_text.strip()
    # If it's just a bare number, use it directly
    if cleaned.isdigit():
        val = int(cleaned)
        if 0 <= val <= 10:
            return val
    # Otherwise scan for first valid number 0-10
    matches = re.findall(r"\b(\d{1,2})\b", cleaned)
    for match in matches:
        val = int(match)
        if 0 <= val <= 10:
            return val
    return FALLBACK_ACTION


def run_episode(
    client: OpenAI,
    task_id: str,
    verbose: bool = False,
) -> dict:
    """
    Run a single episode using the LLM agent.

    Parameters
    ----------
    client : OpenAI
        OpenAI-compatible client.
    task_id : str
        Task to run ("easy", "medium", "hard").
    verbose : bool
        Print per-step details.

    Returns a dict with episode results.
    """
    env = create_env_for_task(task_id, text_mode=True)
    obs = env.reset(seed=42)

    total_reward = 0.0
    trajectory = []
    step = 0

    while not obs.done and step < MAX_STEPS:
        # Build user message from text summary
        user_msg = obs.text_summary if obs.text_summary else obs.message

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
            action_id = parse_action(response)
        except Exception as e:
            print(f"  LLM error at step {step}: {e}", file=sys.stderr)
            action_id = FALLBACK_ACTION
            response = f"ERROR: {e}"

        if verbose:
            print(f"  Step {step}: Action={action_id} ({env.config.action_names[action_id]})")

        action = CropAction(action_id=action_id)
        obs = env.step(action)
        total_reward += obs.reward or 0.0

        trajectory.append({
            "step": step,
            "action_id": action_id,
            "reward": obs.reward,
            "cash": obs.cash_balance,
            "debt": obs.current_debt,
            "soil_n": obs.soil_nitrogen,
            "prices": [
                obs.market_price_crop_1,
                obs.market_price_crop_2,
                obs.market_price_crop_3,
            ]
        })

        step += 1

    # Final state
    final_net_worth = (
        obs.cash_balance - obs.current_debt + obs.soil_nitrogen * 10000
    )
    
    score = grader(task_id, final_net_worth, obs.done and step < MAX_STEPS, trajectory)

    return {
        "task_id": task_id,
        "score": score,
        "steps_completed": step,
        "total_reward": total_reward,
        "final_cash": obs.cash_balance,
        "final_debt": obs.current_debt,
        "final_soil_n": obs.soil_nitrogen,
        "final_net_worth": final_net_worth,
        "bankrupt": obs.done and step < MAX_STEPS,
    }


def main():
    """Run inference on all tasks and print results."""
    print("=" * 60)
    print("CropRL Inference — OpenAI Client")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    results = {}
    for task_id in TASKS:
        print(f"\n--- Task: {task_id} ---")
        print(f"Description: {TASKS[task_id]['description']}")
        result = run_episode(client, task_id, verbose=True)
        results[task_id] = result
        print(f"  Result: {json.dumps(result, indent=2)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tid, r in results.items():
        status = "BANKRUPT" if r["bankrupt"] else "COMPLETED"
        print(
            f"  {tid:10s}: {status:10s} | "
            f"Score: {r['score']:.2f} | "
            f"Steps: {r['steps_completed']:3d} | "
            f"Net Worth: ₹{r['final_net_worth']:,.0f} | "
            f"Total Reward: ₹{r['total_reward']:,.0f}"
        )


if __name__ == "__main__":
    main()
