"""
Ollama Inference Script for CropRL Environment.
===================================
Uses Ollama's OpenAI-compatible API for local LLM testing.
Shows rich per-step details with tqdm progress bar.

Usage:
    # Start Ollama server first
    ollama serve

    # Run inference
    python3 inference_ollama.py
    python3 inference_ollama.py --model qwen3.5:4b --task medium
    python3 inference_ollama.py --model qwen3.5:4b --task easy --verbose
    python3 inference_ollama.py --thinking          # enable extended thinking
    python3 inference_ollama.py --no-thinking        # disable thinking (default)
    python3 inference_ollama.py --all-tasks
"""

import argparse
import json
import sys
import re

from openai import OpenAI
from tqdm import tqdm

from cropRL.config import EnvConfig
from cropRL.models import CroprlAction
from cropRL.tasks import TASKS, create_env_for_task
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import SYSTEM_PROMPT, MAX_STEPS, TEMPERATURE, MAX_TOKENS, FALLBACK_ACTION

# ── Ollama defaults ────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_API_KEY = "ollama"  # placeholder, not actually checked by Ollama

# ── Display helpers ────────────────────────────────────────────
_SEASON_MAP = {
    1: "Winter", 2: "Spring", 3: "Spring", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Winter", 11: "Winter", 12: "Winter",
}
_MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output (qwen3 etc)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_action(response_text: str) -> int:
    """Extract an action integer from the LLM response.
    
    Strips <think> blocks first so reasoning traces don't pollute parsing.
    Then looks for a standalone integer 0-10 in the remaining text.
    """
    cleaned = _strip_thinking(response_text)
    # If the cleaned text is just a number, use it directly
    cleaned_stripped = cleaned.strip()
    if cleaned_stripped.isdigit():
        val = int(cleaned_stripped)
        if 0 <= val <= 10:
            return val
    # Otherwise scan for first valid number
    matches = re.findall(r"\b(\d{1,2})\b", cleaned_stripped)
    for match in matches:
        val = int(match)
        if 0 <= val <= 10:
            return val
    return FALLBACK_ACTION


def _format_cash(v: float) -> str:
    """Format currency with color hint via sign."""
    if v >= 0:
        return f"₹{v:,.0f}"
    return f"-₹{abs(v):,.0f}"


def run_episode_interactive(
    client: OpenAI,
    model_name: str,
    task_id: str,
    verbose: bool = False,
    extra_body: dict | None = None,
    is_qwen: bool = False,
) -> dict:
    """
    Run a single episode with tqdm progress bar and rich per-step output.
    """
    env = create_env_for_task(task_id, text_mode=True)
    obs = env.reset(seed=42)
    config = env.config

    total_reward = 0.0
    trajectory = []
    step = 0

    api_kwargs = {}
    if extra_body:
        api_kwargs["extra_body"] = extra_body

    # Progress bar
    pbar = tqdm(
        total=MAX_STEPS,
        desc=f"  {task_id:8s}",
        bar_format=(
            "  {desc} |{bar:30}| {n_fmt}/{total_fmt} months "
            "[{elapsed}<{remaining}]"
        ),
        leave=True,
    )

    while not obs.done and step < MAX_STEPS:
        obs_text = obs.text_summary if obs.text_summary else obs.message

        # qwen models break with system role — merge into user message.
        # Other models (gemma, llama, etc.) work fine with system role.
        if is_qwen:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n---\n\n{obs_text}"},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]

        response = ""
        action_id = FALLBACK_ACTION
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                **api_kwargs,
            )
            response = completion.choices[0].message.content or ""
            action_id = _parse_action(response)

            # If response was empty, warn and retry once without thinking
            if not response.strip() and extra_body and extra_body.get("think"):
                tqdm.write(f"  [!] Step {step}: Empty response, retrying with think=False...")
                retry = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=min(TEMPERATURE + 0.3, 1.0),
                    max_tokens=MAX_TOKENS,
                    extra_body={"think": False},
                )
                response = retry.choices[0].message.content or ""
                action_id = _parse_action(response)
        except Exception as e:
            tqdm.write(f"  [!] LLM error at step {step}: {e}", file=sys.stderr)
            action_id = FALLBACK_ACTION
            response = ""

        # Execute action
        action = CroprlAction(action_id=action_id)
        obs = env.step(action)
        reward = obs.reward or 0.0
        total_reward += reward

        # Build display values
        month = obs.current_month
        season = _SEASON_MAP.get(month, "?")
        month_str = _MONTH_NAMES[month]
        action_name = config.action_names[action_id]

        crop_type = obs.active_crop_type
        crop_name = config.crop_names[crop_type]
        crop_age = obs.crop_age_months
        if crop_type > 0:
            growth = config.growth_months[crop_type]
            crop_str = f"{crop_name} ({crop_age}/{growth}m)"
        else:
            crop_str = "No crop (Fallow)"

        # Result message
        msg = obs.message.split(" | ")[0] if obs.message else ""
        if len(msg) > 70:
            msg = msg[:67] + "..."

        # ── Verbose block output ──
        if verbose:
            # LLM response line
            if response.strip():
                cleaned = _strip_thinking(response)
                llm_text = cleaned[:30].replace("\n", " ").strip()
            else:
                llm_text = "(empty)"

            reward_str = f"+{reward:.0f}" if reward >= 0 else f"{reward:.0f}"

            tqdm.write(
                f"  {step:2d}. {month_str} {season:<7s} | "
                f"LLM: {llm_text!r:>6s} -> {action_name:<26s} | "
                f"Cash: {_format_cash(obs.cash_balance):<9s} "
                f"Debt: {_format_cash(obs.current_debt):<7s} "
                f"Soil: {obs.soil_nitrogen:.2f}  "
                f"Crop: {crop_str:<16s} | "
                f"R: {reward_str:>6s}"
            )
            # Show environment message on next line if non-trivial
            if msg and action_id != 0:
                tqdm.write(f"      {msg}")

        # Update progress bar postfix with key metrics
        pbar.set_postfix_str(
            f"Cash:{_format_cash(obs.cash_balance)} "
            f"Soil:{obs.soil_nitrogen:.2f} "
            f"{crop_name}",
            refresh=False,
        )
        pbar.update(1)

        trajectory.append({
            "step": step,
            "action_id": action_id,
            "reward": reward,
            "cash": obs.cash_balance,
            "debt": obs.current_debt,
            "soil_n": obs.soil_nitrogen,
        })
        step += 1

    pbar.close()

    # Final stats
    final_net_worth = (
        obs.cash_balance - obs.current_debt + obs.soil_nitrogen * 10000
    )
    bankrupt = obs.done and step < MAX_STEPS

    # Print episode summary
    status = "💀 BANKRUPT" if bankrupt else "✅ COMPLETED"
    print(f"\n  {status} after {step} months")
    print(f"  Final Cash: {_format_cash(obs.cash_balance)}  |  "
          f"Debt: {_format_cash(obs.current_debt)}  |  "
          f"Soil N: {obs.soil_nitrogen:.2f}")
    print(f"  Net Worth: {_format_cash(final_net_worth)}  |  "
          f"Total Reward: {total_reward:+,.0f}")

    return {
        "task_id": task_id,
        "steps_completed": step,
        "total_reward": total_reward,
        "final_cash": obs.cash_balance,
        "final_debt": obs.current_debt,
        "final_soil_n": obs.soil_nitrogen,
        "final_net_worth": final_net_worth,
        "bankrupt": bankrupt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CropRL inference using a local Ollama model."
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_DEFAULT_MODEL,
        help=f"Ollama model name (default: {OLLAMA_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--task",
        default="medium",
        choices=list(TASKS.keys()),
        help="Task difficulty (default: medium)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all tasks instead of a single one",
    )
    parser.add_argument(
        "--base-url",
        default=OLLAMA_BASE_URL,
        help=f"Ollama API base URL (default: {OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step details (action, cash, soil, crop, reward)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment (default: 42)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=None,
        help="Enable extended thinking/reasoning mode (auto-enabled for qwen3 models)",
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable extended thinking mode",
    )
    args = parser.parse_args()

    # Resolve thinking mode
    # IMPORTANT: qwen3 via Ollama's OpenAI API returns EMPTY content with
    # think=True (all tokens consumed by thinking, nothing left for content).
    # Default to False for all models. Users can opt-in with --thinking.
    model_lower = args.model.lower()
    is_qwen = "qwen" in model_lower
    if args.thinking is None:
        args.thinking = False

    if is_qwen and args.thinking:
        print(f"[warn] think=True often causes empty responses with qwen models.")
        print(f"       If you get all-Wait output, try --no-thinking")

    # Build extra_body — only include 'think' for qwen models
    extra_body = {"think": args.thinking} if is_qwen else None

    print("=" * 60)
    print("🌾 CropRL Inference — Ollama (Local)")
    print(f"   Model:    {args.model}")
    print(f"   API:      {args.base_url}")
    print(f"   Thinking: {'✅ enabled' if args.thinking else '❌ disabled'}")
    print("=" * 60)

    client = OpenAI(
        base_url=args.base_url,
        api_key=OLLAMA_API_KEY,
    )

    tasks_to_run = list(TASKS.keys()) if args.all_tasks else [args.task]
    results = {}

    for task_id in tasks_to_run:
        print(f"\n{'─' * 60}")
        print(f"📋 Task: {task_id.upper()}")
        print(f"   {TASKS[task_id]['description']}")
        print(f"{'─' * 60}")
        try:
            result = run_episode_interactive(
                client, args.model, task_id,
                verbose=args.verbose, extra_body=extra_body,
                is_qwen=is_qwen,
            )
            results[task_id] = result
        except Exception as e:
            print(f"  ❌ ERROR: {e}", file=sys.stderr)
            results[task_id] = {"error": str(e)}

    # Final summary table
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("📊 FINAL SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Task':<10s}  {'Status':<12s}  {'Steps':>5s}  "
              f"{'Net Worth':>12s}  {'Reward':>10s}")
        print("  " + "─" * 55)
        for tid, r in results.items():
            if "error" in r:
                print(f"  {tid:<10s}  ❌ ERROR")
            else:
                status = "💀 BANKRUPT" if r["bankrupt"] else "✅ OK"
                print(
                    f"  {tid:<10s}  {status:<12s}  {r['steps_completed']:>5d}  "
                    f"{_format_cash(r['final_net_worth']):>12s}  "
                    f"{r['total_reward']:>+10,.0f}"
                )


if __name__ == "__main__":
    main()
