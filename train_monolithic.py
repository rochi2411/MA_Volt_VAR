"""
train_monolithic.py - Monolithic PPO Training with Multi-Seed Support
======================================================================
Single centralized agent controlling all devices.

Usage:
    python train_monolithic.py --env_name 13Bus --steps 50000 --seed 42
    
"""

import numpy as np
import argparse
import os
import json
import random
from datetime import datetime

# PowerGym
import sys
sys.path.append('./powergym')
from powergym.env_register import make_env

# Gymnasium
import gymnasium as gym
from heuristic_agent import GymnasiumCompatibilityWrapper

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
except ImportError:
    print("Error: stable-baselines3 or torch not installed.")
    exit(1)


# ============================================================
# TRAINING METRICS CALLBACK
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    """Callback to log training metrics for publication-quality plots."""
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        self.rewards_log = []
        
        self._current_ep_reward = 0
        self._current_ep_length = 0
    
    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        self._current_ep_reward += reward
        self._current_ep_length += 1
        
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self._current_ep_reward)
            self.episode_lengths.append(self._current_ep_length)
            self._current_ep_reward = 0
            self._current_ep_length = 0
        
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            recent = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            self.timesteps_log.append(self.num_timesteps)
            self.rewards_log.append(np.mean(recent))
        
        return True
    
    def get_training_data(self) -> dict:
        return {
            "timesteps": self.timesteps_log,
            "rewards": self.rewards_log,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_episodes": len(self.episode_rewards)
        }


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_monolithic(env_name='13Bus', steps=10000, seed=42, model_name=None,
                    save_training_data=True, verbose=1):
    """
    Train Monolithic PPO agent with seed control.
    
    Args:
        env_name: Environment name ('13Bus', '34Bus', '123Bus')
        steps: Total training timesteps
        seed: Random seed for reproducibility
        model_name: Custom model name (auto-generated if None)
        save_training_data: Whether to save training curves
        verbose: Verbosity level
    
    Returns:
        model_path: Path to saved model
        training_data: Dictionary with training metrics
    """
    print(f"\n{'='*60}")
    print(f"Training Monolithic PPO Agent")
    print(f"Environment: {env_name}")
    print(f"Timesteps: {steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # ===== SET ALL RANDOM SEEDS =====
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if torch.backends.mps.is_available():
    #     device = "mps"
    # elif torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "cpu"
    # ===== CREATE ENVIRONMENT =====
    raw_env = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw_env)  # Wrap for compatibility
    
    # Create log directory
    log_dir = f"logs/monolithic_{env_name}_seed{seed}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrap with Monitor
    env = Monitor(env, log_dir + "monitor.csv")
    
    # ===== CREATE MODEL WITH SEED =====
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=verbose,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    
    # ===== CREATE CALLBACK =====
    callback = TrainingMetricsCallback(log_freq=500, verbose=0)
    
    # ===== TRAIN =====
    start_time = datetime.now()
    model.learn(total_timesteps=steps, callback=callback)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # ===== SAVE MODEL =====
    if model_name is None:
        model_name = f"monolithic_agent_{env_name}_seed{seed}"
    
    model_path = f"{model_name}.zip"
    model.save(model_name)
    print(f"\nModel saved: {model_path}")
    
    # ===== SAVE TRAINING DATA =====
    training_data = callback.get_training_data()
    training_data["seed"] = seed
    training_data["env_name"] = env_name
    training_data["total_timesteps"] = steps
    training_data["training_time_seconds"] = training_time
    
    if save_training_data:
        data_path = f"{model_name}_training_data.json"
        with open(data_path, 'w') as f:
            serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in training_data.items()}
            json.dump(serializable, f, indent=2)
        print(f"Training data saved: {data_path}")
    
    # ===== PRINT SUMMARY =====
    final_reward = np.mean(training_data["episode_rewards"][-10:]) if training_data["episode_rewards"] else 0
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Avg Reward (last 10 episodes): {final_reward:.2f}")
    print(f"Total Episodes: {training_data['total_episodes']}")
    print(f"Training Time: {training_time:.1f} seconds")
    print(f"{'='*60}\n")
    
    env.close()
    
    return model_path, training_data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Monolithic PPO Agent')
    
    parser.add_argument('--env_name', type=str, default='13Bus',
                       choices=['13Bus', '34Bus', '123Bus', '8500Node'])
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--no_save_data', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    
    args = parser.parse_args()
    
    train_monolithic(
        env_name=args.env_name,
        steps=args.steps,
        seed=args.seed,
        model_name=args.model_name,
        save_training_data=not args.no_save_data,
        verbose=args.verbose
    )