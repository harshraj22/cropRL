import os
import re
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import wandb

# Ensure the root directory is on the path so cropRL module works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropRL.tasks import create_env_for_task
from cropRL.models import MultiAgentAction
from cropRL.inference import parse_action, get_agent_system_prompt

def get_action_logprobs(model, input_ids, gen_seqs, gen_mask):
    """
    Given full input_ids, generated sequences, and their attention mask,
    compute the sum of log probabilities for the non-padded generated tokens.
    """
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    gen_seq_len = gen_seqs.shape[1]
    gen_logits = logits[:, -gen_seq_len:, :]
    gen_labels = labels[:, -gen_seq_len:]
    
    logprobs = F.log_softmax(gen_logits, dim=-1)
    action_logprobs = logprobs.gather(dim=-1, index=gen_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding tokens
    masked_logprobs = action_logprobs * gen_mask
    return masked_logprobs.sum(dim=-1)

def train(args):
    # Initialize WandB
    wandb.init(project="CropRL-GRPO", name=args.run_name, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # important for batched generation
    
    # Load Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    print("LoRA applied successfully. Trainable parameters:")
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for iteration in range(1, args.num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{args.num_iterations} ---")
        
        # --- 1. Rollout Phase ---
        model.eval() # Prevent dropout noise during rollout
        
        envs = [create_env_for_task(args.task, text_mode=True) for _ in range(args.group_size)]
        
        # Enforce single agent to avoid non-advancing months with static rule agents
        assert envs[0]._ma_cfg.num_agents == 1, "Training script only supports single-agent tasks (e.g., 'easy')."
        
        for env in envs:
            env.reset()
        observations = [env.get_obs(0) for env in envs]
        
        # Get initial net worths for reward shaping
        prev_net_worths = [env._inner.farms[0]._compute_net_worth() for env in envs]
        
        active_envs = list(range(args.group_size))
        trajectories = [[] for _ in range(args.group_size)]
        
        step_count = 0
        with torch.no_grad():
            while active_envs:
                step_count += 1
                prompts = []
                for i in active_envs:
                    obs = observations[i]
                    user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
                    prompt = get_agent_system_prompt(0, 1) + "\n\n" + user_msg + "\nAction:"
                    prompts.append(prompt)
                
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                gen_seqs = outputs[:, inputs.input_ids.shape[1]:]
                action_texts = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
                
                # Mask out right-padding in generation
                gen_mask = (gen_seqs != tokenizer.pad_token_id).long()
                full_seqs = outputs
                old_logprobs = get_action_logprobs(model, full_seqs, gen_seqs, gen_mask)
                
                new_active_envs = []
                for idx, env_idx in enumerate(active_envs):
                    action_text = action_texts[idx]
                    action_id, forum_msg = parse_action(action_text, fallback_action=0)
                    
                    action_obj = MultiAgentAction(action_id=action_id, agent_id=0, forum_message=forum_msg)
                    next_obs = envs[env_idx].step(action_obj)
                    
                    # Reward shaping: Change in exact net worth (including crop/land values)
                    current_net_worth = envs[env_idx]._inner.farms[0]._compute_net_worth()
                    reward = current_net_worth - prev_net_worths[env_idx]
                    prev_net_worths[env_idx] = current_net_worth
                    
                    trajectories[env_idx].append({
                        "input_ids": full_seqs[idx].cpu(),
                        "gen_seqs": gen_seqs[idx].cpu(),
                        "gen_mask": gen_mask[idx].cpu(),
                        "old_logprob": old_logprobs[idx].item(),
                        "reward": reward,
                        "net_worth": current_net_worth,
                        "action_id": action_id
                    })
                    
                    observations[env_idx] = next_obs
                    if not next_obs.done:
                        new_active_envs.append(env_idx)
                        
                active_envs = new_active_envs
        
        # --- 2. Compute Advantages (GRPO) ---
        episode_returns = np.array([sum(step["reward"] for step in traj) for traj in trajectories])
        mean_return = episode_returns.mean()
        std_return = episode_returns.std() + 1e-8
        advantages = (episode_returns - mean_return) / std_return
        
        print(f"Returns: {episode_returns.round(2)}")
        print(f"Mean Return: {mean_return:.2f} | Std: {std_return:.2f}")
        
        # --- 3. Optimization Phase ---
        model.train() # Enable dropout/training mode
        
        # Flatten dataset for randomized mini-batching
        dataset = []
        for env_idx, traj in enumerate(trajectories):
            A_i = advantages[env_idx]
            for step in traj:
                dataset.append({
                    "input_ids": step["input_ids"],
                    "gen_seqs": step["gen_seqs"],
                    "gen_mask": step["gen_mask"],
                    "old_logprob": step["old_logprob"],
                    "A_i": A_i
                })
        
        # Shuffle dataset to break temporal correlations
        random.shuffle(dataset)
        
        total_loss = 0
        total_kl = 0
        optim_steps = 0
        
        optimizer.zero_grad()
        
        # Iterate over steps, accumulating gradients to simulate mini-batches
        for step_idx, step in enumerate(dataset):
            full_seq = step["input_ids"].unsqueeze(0).to(device)
            gen_seqs = step["gen_seqs"].unsqueeze(0).to(device)
            gen_mask = step["gen_mask"].unsqueeze(0).to(device)
            old_logprob = step["old_logprob"]
            A_i = step["A_i"]
            
            # Forward pass current model
            current_logprobs = get_action_logprobs(model, full_seq, gen_seqs, gen_mask).squeeze(0)
            
            # Forward pass reference model (LoRA disabled)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logprobs = get_action_logprobs(model, full_seq, gen_seqs, gen_mask).squeeze(0)
            
            # PPO Ratio
            ratio = torch.exp(current_logprobs - old_logprob)
            
            # KL Divergence Penalty
            kl_div = torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1
            
            # Clipped Surrogate Objective
            surr1 = ratio * A_i
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A_i
            policy_loss = -torch.min(surr1, surr2)
            
            loss = policy_loss + args.beta * kl_div
            
            # Gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_kl += kl_div.item()
            
            # Step optimizer periodically
            if (step_idx + 1) % args.gradient_accumulation_steps == 0 or (step_idx + 1) == len(dataset):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                optim_steps += 1
                
        # Logging
        avg_loss = total_loss / max(1, optim_steps)
        avg_kl = total_kl / max(1, optim_steps)
        
        wandb.log({
            "iteration": iteration,
            "mean_return": mean_return,
            "std_return": std_return,
            "loss": avg_loss,
            "kl_divergence": avg_kl,
            "max_return": episode_returns.max(),
            "min_return": episode_returns.min(),
        })
        
        # Save Checkpoint
        if iteration % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{iteration}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")

    print("Training complete!")
    model.save_pretrained(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model path")
    parser.add_argument("--run_name", type=str, default="CropRL_GRPO_Run_1", help="WandB run name")
    parser.add_argument("--task", type=str, default="easy", help="CropRL task identifier")
    parser.add_argument("--num_iterations", type=int, default=50, help="Total training iterations")
    parser.add_argument("--group_size", type=int, default=8, help="Number of trajectories to collect per iteration (G)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Batch size equivalent via grad accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for LoRA")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=15, help="Max tokens per action generation")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N iterations")
    parser.add_argument("--output_dir", type=str, default="./train/checkpoints", help="Output directory for checkpoints")
    
    args = parser.parse_args()
    train(args)
