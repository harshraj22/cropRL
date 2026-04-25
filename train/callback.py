from transformers import TrainerCallback
import os
import subprocess

class SaveToStorageCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def on_save(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        # Sync to HuggingFace bucket using subprocess
        try:
            print(f"Syncing epoch {epoch} checkpoint to HuggingFace bucket...")
            subprocess.run(
                [
                    "hf", "buckets", "sync",
                    "--exclude", ".*",
                    self.output_dir,
                    "hf://buckets/Utkarshp1/croprl-workspace/sft_checkpoints/colab"
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Sync complete for epoch {epoch}!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to sync checkpoint. Error: {e.stderr}")