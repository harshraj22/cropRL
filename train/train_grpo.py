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

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model

import bitsandbytes as bnb

import wandb

# Ensure the root directory is on the path so cropRL module works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropRL.tasks import create_env_for_task
from cropRL.models import MultiAgentAction
from cropRL.inference import parse_action, get_agent_system_prompt

def collate_batch(steps, device):
    # Pad sequences to same length within batch
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [s["input_ids"] for s in steps], batch_first=True
    ).to(device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [s["attention_mask"] for s in steps], batch_first=True
    ).to(device)
    gen_seqs = torch.nn.utils.rnn.pad_sequence(
        [s["gen_seqs"] for s in steps], batch_first=True
    ).to(device)
    gen_mask = torch.nn.utils.rnn.pad_sequence(
        [s["gen_mask"] for s in steps], batch_first=True
    ).to(device)
    old_logprobs = torch.tensor([s["old_logprob"] for s in steps], device=device)
    advantages = torch.tensor([s["A_i"] for s in steps], device=device)
    return input_ids, attention_mask, gen_seqs, gen_mask, old_logprobs, advantages


def get_action_logprobs(model, input_ids, attention_mask, gen_seqs, gen_mask):
    """
    Given full input_ids, their attention mask, generated sequences, and their mask,
    compute the sum of log probabilities for the non-padded generated tokens.
    """
    outputs = model(input_ids, attention_mask=attention_mask)
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

get_action_logprobs = torch.compile(get_action_logprobs, mode="reduce-overhead")

def get_action_prefix_fn(tokenizer, prompt_length):
    """Creates a prefix_allowed_tokens_fn to constrain generation to valid action formats."""
    digit_tokens = {str(i): tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)}
    space_token = tokenizer.encode(" ", add_special_tokens=False)[0]
    
    token_1 = digit_tokens["1"]
    tokens_0_to_4 = [digit_tokens[str(i)] for i in range(5)]
    all_digits = list(digit_tokens.values())
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        gen_tokens = input_ids[prompt_length:]
        if len(gen_tokens) == 0:
            return all_digits
        elif len(gen_tokens) == 1:
            first = gen_tokens[0].item()
            if first == token_1:
                return tokens_0_to_4 + [space_token, tokenizer.eos_token_id]
            else:
                return [space_token, tokenizer.eos_token_id]
        elif len(gen_tokens) == 2:
            first = gen_tokens[0].item()
            second = gen_tokens[1].item()
            if first == token_1 and second == token_1: # "11"
                return [space_token] # Force space after "11"
            else:
                return [tokenizer.eos_token_id]
        else:
            first = gen_tokens[0].item()
            if len(gen_tokens) > 1:
                second = gen_tokens[1].item()
                if first == token_1 and second == token_1:
                    return list(range(tokenizer.vocab_size))
            return [tokenizer.eos_token_id]
            
    return prefix_allowed_tokens_fn

def train(args):
    print("="*50)
    print("GRPO TRAINING CONFIGURATION")
    print(f"Model Taken From:    {args.model_name}")
    import os
    model_source = "Local Checkpoint" if os.path.isdir(args.model_name) else "HuggingFace Hub"
    print(f"Model Source:        {model_source}")
    print(f"Task:                {args.task}")
    print(f"Group Size (G):      {args.group_size}")
    print(f"LoRA Targets:        ['q_proj', 'v_proj']")
    print("="*50)
    
    # Initialize WandB
    wandb.init(project="CropRL-GRPO", name=args.run_name, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        
    tokenizer.padding_side = "left" # important for batched generation
    
    # Load Model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            model.get_input_embeddings().weight[-1].zero_()
    
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
    model.gradient_checkpointing_enable()  

    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_iterations,
        num_training_steps=args.num_iterations
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Training Iterations"):
        print(f"\n--- Iteration {iteration}/{args.num_iterations} ---")
        
        # --- 1. Rollout Phase ---
        model.eval() # Prevent dropout noise during rollout
        
        envs = [create_env_for_task(args.task, text_mode=True) for _ in range(args.group_size)]
        n_agents = envs[0]._ma_cfg.num_agents
        
        # Curriculum Learning: Expanding horizon starts small to learn short-term consequences first
        current_max_months = min(60, 10 + iteration * 2)
        print(f"Curriculum Horizon: {current_max_months} months")
        
        group_seed = iteration
        for env_idx, env in enumerate(envs):
            env._env_cfg.max_months = current_max_months
            
            env.reset(seed=group_seed)
        
        # Get initial net worths for reward shaping (per env, per agent)
        prev_net_worths = [[env._farms[a].compute_net_worth() for a in range(n_agents)] for env in envs]
        
        active_envs = list(range(args.group_size))
        done_agents = {i: set() for i in range(args.group_size)}
        histories = {i: {a: [] for a in range(n_agents)} for i in range(args.group_size)}
        trajectories = [[[] for _ in range(n_agents)] for _ in range(args.group_size)]
        
        step_count = 0
        total_env_steps = envs[0]._env_cfg.max_months * envs[0]._ma_cfg.action_slots_per_month
        with torch.no_grad(), tqdm(total=total_env_steps, desc="Rollout Phase", leave=False) as pbar:
            while active_envs:
                step_count += 1
                # Use the rotating turn order from the first active env (valid proxy for batch)
                for agent_slot in range(n_agents):
                    prompts = []
                    valid_env_indices = []
                    agent_ids_for_batch = []
                    
                    # Fetch fresh observations for this agent across active environments
                    for env_idx in active_envs:
                        turn_order = envs[env_idx].get_turn_order()
                        agent_id = turn_order[agent_slot]
                        if agent_id in done_agents[env_idx]:
                            action_obj = MultiAgentAction(action_id=0, agent_id=agent_id, forum_message=None)
                            envs[env_idx].step(action_obj)
                            continue

                        obs = envs[env_idx].get_obs(agent_id)
                        
                        if obs.done:
                            done_agents[env_idx].add(agent_id)
                            # Dead/done agents must wait out their slots so they don't block TimeController
                            action_obj = MultiAgentAction(action_id=0, agent_id=agent_id, forum_message=None)
                            envs[env_idx].step(action_obj)
                            continue
                            
                        user_msg = obs.text_summary if getattr(obs, "text_summary", None) else str(obs)
                        history_block = "\n".join(histories[env_idx][agent_id][-12:]) if histories[env_idx][agent_id] else "None"
                        user_msg += f"\n\nRecent History:\n{history_block}"

                        messages = [
                            {"role": "system", "content": get_agent_system_prompt(agent_id, n_agents)},
                            {"role": "user", "content": user_msg}
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False,
                            enable_thinking=False
                        )

                        prompts.append(prompt)
                        valid_env_indices.append(env_idx)
                        agent_ids_for_batch.append(agent_id)
                    
                    if not prompts:
                        continue
                        
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                    
                    prefix_fn = get_action_prefix_fn(tokenizer, inputs.input_ids.shape[1])
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.8,
                        top_k=20,
                        min_p=0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        prefix_allowed_tokens_fn=prefix_fn,
                    )

                    # Mask out right-padding in generation
                    full_seqs = outputs

                    prompt_len = inputs.input_ids.shape[1]
                    gen_seqs = outputs[:, prompt_len:]
                    action_texts = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)

                    gen_mask = (gen_seqs != tokenizer.pad_token_id).long() 
                    full_attention_mask = torch.cat([inputs.attention_mask, gen_mask], dim=1)
                    

                    old_logprobs = get_action_logprobs(model, full_seqs, full_attention_mask, gen_seqs, gen_mask)
                    
                    for idx, env_idx in enumerate(valid_env_indices):
                        agent_id = agent_ids_for_batch[idx]
                        action_text = action_texts[idx]
                        act_id, forum_msg = parse_action(action_text, fallback_action=0)
                        
                        action_obj = MultiAgentAction(action_id=act_id, agent_id=agent_id, forum_message=forum_msg)
                        next_obs = envs[env_idx].step(action_obj)
                        
                        # Reward shaping: Change in exact net worth (including crop/land values)
                        current_net_worth = envs[env_idx]._farms[agent_id].compute_net_worth()
                        reward = current_net_worth - prev_net_worths[env_idx][agent_id]
                        prev_net_worths[env_idx][agent_id] = current_net_worth
                        
                        action_name = envs[env_idx]._env_cfg.action_names[act_id] if act_id < len(envs[env_idx]._env_cfg.action_names) else f"Action {act_id}"
                        histories[env_idx][agent_id].append(f"Step {getattr(next_obs, 'current_step', step_count)}: Selected '{action_name}' -> Reward {reward:+.2f}")
                        
                        trajectories[env_idx][agent_id].append({
                            "input_ids": full_seqs[idx].cpu(),
                            "attention_mask": full_attention_mask[idx].cpu(),
                            "gen_seqs": gen_seqs[idx].cpu(),
                            "gen_mask": gen_mask[idx].cpu(),
                            "old_logprob": old_logprobs[idx].item(),
                            "reward": reward,
                            "action_id": act_id
                        })
                        
                        if next_obs.done:
                            done_agents[env_idx].add(agent_id)
                            
                # Update active envs list (only keep envs where not all agents are done)
                active_envs = [i for i in active_envs if len(done_agents[i]) < n_agents]
                pbar.update(1)
        
        # --- 2. Compute Advantages (GRPO) ---
        # Normalize returns across all agents and all group environments
        all_returns = []
        for env_idx in range(args.group_size):
            for agent_id in range(n_agents):
                ret = sum(step["reward"] for step in trajectories[env_idx][agent_id])
                all_returns.append(ret)
                
        all_returns = np.array(all_returns)
        mean_return = all_returns.mean()
        std_return = all_returns.std() + 1e-8
        
        print(f"Returns: {all_returns.round(2)}")
        print(f"Mean Return: {mean_return:.2f} | Std: {std_return:.2f}")
        
        # --- 3. Optimization Phase ---
        model.train() # Enable dropout/training mode
        
        # Flatten dataset for randomized mini-batching
        dataset = []
        ret_idx = 0
        for env_idx in range(args.group_size):
            for agent_id in range(n_agents):
                A_i = (all_returns[ret_idx] - mean_return) / std_return
                for step in trajectories[env_idx][agent_id]:
                    dataset.append({
                        "input_ids": step["input_ids"],
                        "attention_mask": step["attention_mask"],
                        "gen_seqs": step["gen_seqs"],
                        "gen_mask": step["gen_mask"],
                        "old_logprob": step["old_logprob"],
                        "A_i": A_i
                    })
                ret_idx += 1
        
        # Shuffle dataset to break temporal correlations
        random.shuffle(dataset)
        
        total_loss = 0
        total_kl = 0
        optim_steps = 0
        
        optimizer.zero_grad()
        
        # Iterate over steps, accumulating gradients to simulate mini-batches
        # Iterate over mini-batches
        MINI_BATCH_SIZE = 16  # tune to VRAM
        for batch_start in tqdm(range(0, len(dataset), MINI_BATCH_SIZE), desc="Optimization Phase", leave=False):
            batch = dataset[batch_start:batch_start + MINI_BATCH_SIZE]
            input_ids, attn_mask, gen_seqs, gen_mask, old_lp, A_i = collate_batch(batch, device)

            # Forward pass current model
            current_logprobs = get_action_logprobs(model, input_ids, attn_mask, gen_seqs, gen_mask)

            # Forward pass reference model (LoRA disabled)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logprobs = get_action_logprobs(model, input_ids, attn_mask, gen_seqs, gen_mask)

            # PPO Ratio
            ratio = torch.exp(current_logprobs - old_lp)

            # KL Divergence Penalty
            kl_div = torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1

            # Clipped Surrogate Objective
            surr1 = ratio * A_i
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * A_i
            loss = (-torch.min(surr1, surr2) + args.beta * kl_div).mean()

            loss.backward()

            total_loss += loss.item() * len(batch)
            total_kl += kl_div.mean().item() * len(batch)
            optim_steps += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        avg_loss = total_loss / max(1, len(dataset))
        avg_kl = total_kl / max(1, len(dataset))
        
        wandb.log({
            "iteration": iteration,
            "mean_return": mean_return,
            "mean_return_per_month": mean_return / max(1, current_max_months),
            "current_horizon": current_max_months,
            "dataset_size": len(dataset),
            "std_return": std_return,
            "loss": avg_loss,
            "kl_divergence": avg_kl,
            "max_return": all_returns.max(),
            "min_return": all_returns.min(),
            "learning_rate": lr_scheduler.get_last_lr()[0],
        })
        
        # Step the learning rate scheduler at the end of each iteration
        lr_scheduler.step()
        
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Hugging Face model path")
    parser.add_argument("--run_name", type=str, default="CropRL_GRPO_Run_1", help="WandB run name")
    parser.add_argument("--task", type=str, default="easy_2agent", help="CropRL task identifier")
    parser.add_argument("--num_iterations", type=int, default=50, help="Total training iterations")
    parser.add_argument("--group_size", type=int, default=8, help="Number of trajectories to collect per iteration (G)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Batch size equivalent via grad accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for LoRA")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Scheduler type (cosine, linear)")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens per action generation")
    parser.add_argument("--save_every", type=int, default=2, help="Save checkpoint every N iterations")
    parser.add_argument("--output_dir", type=str, default="./train/checkpoints", help="Output directory for checkpoints")
    
    args = parser.parse_args()
    train(args)
