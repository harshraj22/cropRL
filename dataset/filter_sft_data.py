import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_all_records(input_path):
    records = []
    with open(input_path, "r") as fin:
        for line in tqdm(fin, desc="Loading"):
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_episode_final_returns(records):
    """
    For each (episode, agent_id), find the total_return at the final step.
    The final step is the one with the highest total_steps value.
    This is the true episode-level return.
    """
    # (episode, agent_id) -> (max_total_steps_seen, total_return_at_that_step)
    episode_final = {}
    for record in records:
        meta = record.get("metadata", {})
        key = (meta.get("episode"), meta.get("agent_id"))
        total_steps = meta.get("total_steps", 0)
        total_return = meta.get("total_return", 0.0)

        if key not in episode_final or total_steps > episode_final[key][0]:
            episode_final[key] = (total_steps, total_return)

    # Return just the final return per key
    return {key: val[1] for key, val in episode_final.items()}


def filter_by_episode(records, min_reward, episode_final_returns):
    """
    Keep ALL steps of an episode if its final total_return >= min_reward.
    Simplest and most consistent — good episodes have good actions throughout.
    """
    kept = []
    for record in records:
        meta = record.get("metadata", {})
        key = (meta.get("episode"), meta.get("agent_id"))
        if episode_final_returns.get(key, float("-inf")) >= min_reward:
            kept.append(record)
    return kept


def filter_by_future_return(records, min_reward, episode_final_returns):
    """
    Keep a step only if its future return G_t >= min_reward.
    G_t = final_total_return - total_return[t]
    
    This filters at step granularity: a good action in a bad episode can still
    be kept if the future from that point was profitable.
    """
    kept = []
    for record in records:
        meta = record.get("metadata", {})
        key = (meta.get("episode"), meta.get("agent_id"))
        total_return_t = meta.get("total_return", 0.0)
        final_return = episode_final_returns.get(key, float("-inf"))

        # G_t: return earned from step t onwards
        g_t = final_return - total_return_t
        if g_t >= min_reward:
            kept.append(record)
    return kept


def print_action_distribution(records, label=""):
    dist = defaultdict(int)
    for record in records:
        messages = record.get("messages", [])
        if messages and messages[-1]["role"] == "assistant":
            action = messages[-1]["content"].strip().split()[0]  # handles "11 <msg>" too
            dist[action] += 1
    total = sum(dist.values())
    print(f"\nAction distribution ({label}):")
    for action in sorted(dist, key=lambda x: int(x) if x.isdigit() else 99):
        count = dist[action]
        bar = "█" * int(30 * count / total)
        print(f"  Action {action:>2}: {bar} {count} ({100*count/total:.1f}%)")


def main(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading records from {input_path}...")
    records = load_all_records(input_path)
    print(f"Total records loaded: {len(records)}")

    print("\nComputing per-episode final returns...")
    episode_final_returns = compute_episode_final_returns(records)
    print(f"Unique (episode, agent) trajectories found: {len(episode_final_returns)}")

    final_returns = list(episode_final_returns.values())
    print(f"Episode return stats:")
    print(f"  Min:    {min(final_returns):.1f}")
    print(f"  Max:    {max(final_returns):.1f}")
    print(f"  Mean:   {sum(final_returns)/len(final_returns):.1f}")
    sorted_returns = sorted(final_returns)
    p30 = sorted_returns[int(0.30 * len(sorted_returns))]
    p50 = sorted_returns[int(0.50 * len(sorted_returns))]
    p70 = sorted_returns[int(0.70 * len(sorted_returns))]
    print(f"  P30:    {p30:.1f}")
    print(f"  P50:    {p50:.1f}")
    print(f"  P70:    {p70:.1f}")
    print(f"\n  (Hint: for top-30%% filtering, set --min_reward {p70:.0f})")

    print(f"\nFiltering mode: {args.mode} | min_reward threshold: {args.min_reward}")

    if args.mode == "episode":
        kept = filter_by_episode(records, args.min_reward, episode_final_returns)
    elif args.mode == "future_return":
        kept = filter_by_future_return(records, args.min_reward, episode_final_returns)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print_action_distribution(records, label="before filtering")
    print_action_distribution(kept, label="after filtering")

    kept_keys = set()
    for record in kept:
        meta = record.get("metadata", {})
        kept_keys.add((meta.get("episode"), meta.get("agent_id")))

    print(f"\n--- Filtering Complete ---")
    print(f"Total rows evaluated:              {len(records)}")
    print(f"Total rows kept:                   {len(kept)} ({100*len(kept)/max(1,len(records)):.1f}%)")
    print(f"Unique (ep, agent) kept:           {len(kept_keys)}")
    print(f"Saving to {output_path}...")

    with open(output_path, "w") as fout:
        for record in kept:
            fout.write(json.dumps(record) + "\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SFT JSONL dataset by return quality.")
    parser.add_argument("--input_file", type=str, default="dataset/sft_data.jsonl")
    parser.add_argument("--output_file", type=str, default="dataset/sft_data_filtered.jsonl")
    parser.add_argument(
        "--mode",
        type=str,
        default="episode",
        choices=["episode", "future_return"],
        help=(
            "episode: keep all steps of episodes whose FINAL return >= min_reward. "
            "future_return: keep step t only if G_t = final_return - total_return[t] >= min_reward."
        ),
    )
    parser.add_argument(
        "--min_reward",
        type=float,
        default=0.0,
        help="Threshold. Run once without filtering to see P30/P50/P70 stats, then set this."
    )
    args = parser.parse_args()
    main(args)