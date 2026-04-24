import os
import sys
import argparse
import torch
from pathlib import Path
from datasets import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import wandb

# Ensure the root directory is on the path so cropRL module works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cropRL.inference import get_agent_system_prompt

def create_dummy_dataset():
    """
    Creates an in-memory Hugging Face Dataset with a single sample data point
    formatted as a list of conversational messages.
    """
    system_prompt = get_agent_system_prompt(agent_id=0, num_agents=4)
    
    # A dummy observation from the environment
    user_observation = "Month 1. Cash: 1000. Land: Fallow. Soil N: 1.0. Prices: Wheat=100, Corn=150."
    
    # A dummy ideal action (e.g., Plant Corn)
    ideal_response = "1"
    
    sample = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_observation},
            {"role": "assistant", "content": ideal_response}
        ]
    }
    
    return Dataset.from_list([sample])


def train(args):
    # Initialize WandB
    wandb.init(project="CropRL-SFT", name=args.run_name, config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    # --- 2. Apply LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # --- 3. Load Dataset ---
    print("Loading dataset...")
    # For a real pipeline, you would load from disk:
    # dataset = load_dataset("json", data_files="data.jsonl", split="train")
    dataset = create_dummy_dataset()
    
    # --- 4. Configure SFTTrainer ---
    print("Configuring SFTTrainer...")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_steps=args.save_every,
        max_seq_length=args.max_seq_length,
        report_to="wandb"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # --- 5. Execute Training ---
    print("Starting training...")
    trainer.train()
    
    print("Training complete! Merging LoRA weights into base model...")
    # Merge the trained LoRA adapter into the base model
    model = trainer.model.merge_and_unload()
    
    # Save the fully merged model (which can now be used as the base model for GRPO)
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Merged model saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General & Architecture (matching GRPO)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Instruct", help="Hugging Face model path")
    parser.add_argument("--run_name", type=str, default="CropRL_SFT_Run_1", help="WandB run name")
    
    # Training Hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Grad accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for LoRA")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Scheduler type (cosine, linear)")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    
    # LoRA Config
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # Output
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./train/sft_checkpoints", help="Output directory")
    
    args = parser.parse_args()
    train(args)
