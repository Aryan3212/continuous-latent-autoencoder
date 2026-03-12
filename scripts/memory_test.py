import torch
import yaml
from utils.config import load_config, apply_overrides
import sys

def main():
    cfg = load_config("configs/exp0.yaml")
    
    # Let's import the train logic roughly or just run the training script for a few steps
    # to see peak memory.
    import time
    import subprocess
    
    # We will invoke train.py but set max_steps to 50
    print("Running dry-run memory test with train.py...")
    cmd = [
        sys.executable, "train.py", 
        "--config", "configs/exp0.yaml", 
        "--max_steps", "50",
        "--log_interval_steps", "10",
        "--save_interval_steps", "50"
    ]
    
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["WANDB_MODE"] = "disabled"
    
    subprocess.run(cmd, env=env)
    print("Done dry-run.")

if __name__ == "__main__":
    import os
    main()
