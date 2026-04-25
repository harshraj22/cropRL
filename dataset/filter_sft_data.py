import json
import argparse
from pathlib import Path
from tqdm import tqdm

def main(args):
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    kept_lines = 0
    
    # We will keep track of how many unique episodes/agents were kept
    kept_agents_per_episode = set()
    
    print(f"Filtering {input_path} for total_return >= {args.min_reward}...")
    
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc="Filtering Dataset"):
            if not line.strip():
                continue
                
            total_lines += 1
            data = json.loads(line)
            
            # Check the metadata for total_return
            metadata = data.get("metadata", {})
            total_return = metadata.get("total_return", 0.0)
            
            if total_return >= args.min_reward:
                fout.write(line)
                kept_lines += 1
                
                ep = metadata.get("episode")
                agent = metadata.get("agent_id")
                if ep is not None and agent is not None:
                    kept_agents_per_episode.add((ep, agent))
                    
    print("\n--- Filtering Complete ---")
    print(f"Total rows evaluated:  {total_lines}")
    print(f"Total rows kept:       {kept_lines} ({kept_lines/max(1, total_lines)*100:.1f}%)")
    print(f"Total unique (ep, agent) trajectories kept: {len(kept_agents_per_episode)}")
    print(f"Filtered dataset saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter SFT JSONL dataset by minimum total return.")
    parser.add_argument("--input_file", type=str, default="dataset/sft_data.jsonl", help="Input JSONL path")
    parser.add_argument("--output_file", type=str, default="dataset/sft_data_filtered.jsonl", help="Output JSONL path")
    parser.add_argument("--min_reward", type=float, default=1000.0, help="Minimum total_return to keep the trajectory")
    args = parser.parse_args()
    main(args)
