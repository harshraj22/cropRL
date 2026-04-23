import os
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

import wandb

# Ensure the root directory is on the path so cropRL module works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropRL.tasks import create_env_for_task
from cropRL.models import MultiAgentAction
from cropRL.inference import parse_action, get_agent_system_prompt

def get_action_logprobs(model, input_ids, gen_seq_len):
    """
    Given full input_ids (prompt + generated_sequence), compute the log probabilities
    of the generated sequence tokens.
    """
    outputs = model(input_ids)
    # logits shape: (batch_size, seq_len, vocab_size)
    # We want the logits corresponding to the generated sequence.
    # The generated sequence starts at index: seq_len - gen_seq_len
    # For a token at index i, its logit is at index i-1.
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    
    # Extract only the generated part
    gen_logits = logits[:, -gen_seq_len:, :]
    gen_labels = labels[:, -gen_seq_len:]
    
    logprobs = F.log_softmax(gen_logits, dim=-1)
    action_logprobs = logprobs.gather(dim=-1, index=gen_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum logprobs over the generated sequence
    return action_logprobs.sum(dim=-1)

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
    model.train()
    print("LoRA applied successfully. Trainable parameters:")
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for iteration in range(1, args.num_iterations + 1):
        print(f"\n--- Iteration {iteration}/{args.num_iterations} ---")
        
        # 1. Rollout Phase (Group size G)
        envs = [create_env_for_task(args.task, text_mode=True) for _ in range(args.group_size)]
        observations = [env.reset() for env in envs]
        # In single agent task, env.reset() returns a dict or list? Wait. 
        # Actually `env.reset()` returns dict of observations in old architecture, 
        # but in new architecture, it just resets. We must call get_obs(0).
        observations = [env.get_obs(0) for env in envs]
        
        active_envs = list(range(args.group_size))
        
        # Store rollout data: list of lists
        trajectories = [[] for _ in range(args.group_size)]
        
        step_count = 0
        with torch.no_grad():
            while active_envs:
                step_count += 1
                prompts = []
                for i in active_envs:
                    obs = observations[i]
                    # Format prompt
                    user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
                    prompt = get_agent_system_prompt(0, 1) + "\n\n" + user_msg + "\nAction:"
                    prompts.append(prompt)
                
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                
                # Generate actions
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Process outputs
                gen_seqs = outputs[:, inputs.input_ids.shape[1]:]
                action_texts = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
                
                # Calculate old logprobs for PPO/GRPO reference
                full_seqs = outputs
                gen_seq_len = gen_seqs.shape[1]
                old_logprobs = get_action_logprobs(model, full_seqs, gen_seq_len)
                
                new_active_envs = []
                for idx, env_idx in enumerate(active_envs):
                    action_text = action_texts[idx]
                    action_id, forum_msg = parse_action(action_text, fallback_action=0)
                    
                    action_obj = MultiAgentAction(action_id=action_id, agent_id=0, forum_message=forum_msg)
                    next_obs = envs[env_idx].step(action_obj)
                    
                    # Reward shaping: Change in net worth
                    reward = next_obs.reward or 0.0
                    
                    trajectories[env_idx].append({
                        "input_ids": full_seqs[idx].cpu(),
                        "gen_seq_len": gen_seq_len,
                        "old_logprob": old_logprobs[idx].item(),
                        "reward": reward,
                        "net_worth": next_obs.cash_balance + next_obs.current_debt, # We'll track final later
                        "action_id": action_id
                    })
                    
                    observations[env_idx] = next_obs
                    
                    if not next_obs.done:
                        new_active_envs.append(env_idx)
                        
                active_envs = new_active_envs
        
        # 2. Compute Advantages (GRPO Normalization)
        episode_returns = np.array([sum(step["reward"] for step in traj) for traj in trajectories])
        mean_return = episode_returns.mean()
        std_return = episode_returns.std() + 1e-8
        advantages = (episode_returns - mean_return) / std_return
        
        print(f"Returns: {episode_returns.round(2)}")
        print(f"Mean Return: {mean_return:.2f} | Std: {std_return:.2f}")
        
        # 3. Optimization Phase
        model.train()
        total_loss = 0
        total_kl = 0
        optim_steps = 0
        
        # For each trajectory and step, compute GRPO loss
        for env_idx, traj in enumerate(trajectories):
            A_i = advantages[env_idx]
            
            for step in traj:
                full_seq = step["input_ids"].unsqueeze(0).to(device)
                gen_seq_len = step["gen_seq_len"]
                old_logprob = step["old_logprob"]
                
                # Forward pass current model
                current_logprobs = get_action_logprobs(model, full_seq, gen_seq_len).squeeze(0)
                
                # Forward pass reference model (LoRA disabled)
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logprobs = get_action_logprobs(model, full_seq, gen_seq_len).squeeze(0)
                
                # PPO Ratio
                ratio = torch.exp(current_logprobs - old_logprob)
                
                # KL Divergence Penalty (approx)
                kl_div = torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1
                
                # Clipped Surrogate Objective
                surr1 = ratio * A_i
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A_i
                policy_loss = -torch.min(surr1, surr2)
                
                # Total Loss
                loss = policy_loss + args.beta * kl_div
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                
                total_loss += loss.item()
                total_kl += kl_div.item()
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
