import os
import sys
import json
import argparse
import torch
from pathlib import Path

# Ensure the root directory is on the path so cropRL module works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from cropRL.tasks import create_env_for_task
from cropRL.models import MultiAgentAction
from cropRL.inference import get_agent_system_prompt, parse_action
from tqdm import tqdm

def main(args):
    # Ensure output dir exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Teacher Model: {args.model_name} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Run episodes
    total_samples = 0
    with open(output_path, "a") as f:
        for ep in tqdm(range(1, args.num_episodes + 1), desc="Generating SFT Data"):
            env = create_env_for_task(args.task, text_mode=True)
            # Use a different seed per episode
            env.reset(seed=args.seed_base + ep)
            
            n = env._ma_cfg.num_agents
            max_steps = env._env_cfg.max_steps * n
            
            done_agents = set()
            histories = {i: [] for i in range(n)}
            agent_rewards = {i: 0.0 for i in range(n)}
            episode_buffer = {i: [] for i in range(n)}
            total_steps = 0
            
            while len(done_agents) < n and total_steps < max_steps:
                for agent_id in env.get_turn_order():
                    obs = env.get_obs(agent_id)
                    
                    if obs.done:
                        done_agents.add(agent_id)
                        action_id = 0
                        forum_message = None
                    else:
                        # Prepare the input we give to the model
                        user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
                        history_block = "\n".join(histories[agent_id][-12:]) if histories[agent_id] else "None"
                        user_msg += f"\n\nRecent History:\n{history_block}"
                        
                        system_prompt = get_agent_system_prompt(agent_id, n)
                        
                        # Generate action from local HF model
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=20,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id
                            )
                        
                        # Extract the newly generated tokens
                        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        
                        action_id, forum_message = parse_action(response_text, fallback_action=0)
                        
                        # Format the target string (the ideal response we want the small model to learn)
                        if action_id == 11 and forum_message:
                            target_response = f"{action_id} {forum_message}"
                        else:
                            target_response = str(action_id)
                        
                        # Construct the Hugging Face standard 'messages' dict
                        data_point = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_msg},
                                {"role": "assistant", "content": target_response}
                            ],
                            # Extra metadata for potential later filtering
                            "metadata": {
                                "episode": ep,
                                "agent_id": agent_id,
                                "total_steps": total_steps,
                                "action_id": action_id,
                            }
                        }
                        # Buffer the data point instead of writing immediately
                        episode_buffer[agent_id].append(data_point)
                        
                    action_name = env._env_cfg.action_names[action_id] if action_id < len(env._env_cfg.action_names) else f"Action {action_id}"
                    action = MultiAgentAction(action_id=action_id, agent_id=agent_id, forum_message=forum_message)
                    new_obs = env.step(action)
                    
                    reward = new_obs.reward or 0.0
                    total_steps += 1
                    
                    histories[agent_id].append(f"Step {getattr(new_obs, 'current_step', total_steps)}: Selected '{action_name}' -> Reward {reward:+.2f}")
                    agent_rewards[agent_id] += reward
                    
                    if new_obs.done:
                        done_agents.add(agent_id)
            
            # End of episode: write buffered data with total_return
            for agent_id, data_points in episode_buffer.items():
                total_return = agent_rewards[agent_id]
                for dp in data_points:
                    dp["metadata"]["total_return"] = total_return
                    f.write(json.dumps(dp) + "\n")
                    total_samples += 1
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Instruct", help="HuggingFace Teacher Model")
    parser.add_argument("--task", type=str, default="easy", help="CropRL task identifier")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of full episodes to run")
    parser.add_argument("--output_file", type=str, default="dataset/sft_data.jsonl", help="Output JSONL path")
    parser.add_argument("--seed_base", type=int, default=1000, help="Base seed for environment")
    args = parser.parse_args()
    main(args)
