"""
train_marl.py - Multi-Agent (IPPO) Training with Multi-Seed Support
=====================================================================
Modified to support:
- Reproducible training with seed control
- Training curve logging
- Proper model naming

Usage:
    python train_marl.py --env_name 13Bus --steps 50000 --seed 42
    python train_marl.py --env_name 34Bus --steps 100000 --seed 123
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from powergym.ma_env import MultiAgentPowerGrid
import argparse
import os
import json
import random
from datetime import datetime

# Stable-Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
except ImportError:
    print("Error: stable-baselines3 or torch not installed.")
    print("Run: pip install stable-baselines3 shimmy torch")
    exit(1)


# ============================================================
# TRAINING METRICS CALLBACK (NEW)
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    """
    Callback to log training metrics for publication-quality plots.
    Records episode rewards at regular intervals.
    """
    
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Logging at intervals
        self.timesteps_log = []
        self.rewards_log = []
        
        # Current episode accumulators
        self._current_ep_reward = 0
        self._current_ep_length = 0
    
    def _on_step(self) -> bool:
        # Accumulate reward for current episode
        reward = self.locals.get('rewards', [0])[0]
        self._current_ep_reward += reward
        self._current_ep_length += 1
        
        # Check for episode end
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self._current_ep_reward)
            self.episode_lengths.append(self._current_ep_length)
            
            # Reset accumulators
            self._current_ep_reward = 0
            self._current_ep_length = 0
        
        # Log at intervals (for plotting)
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            # Average over last 10 episodes
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            
            self.timesteps_log.append(self.num_timesteps)
            self.rewards_log.append(avg_reward)
            
            if self.verbose > 0:
                print(f"  Step {self.num_timesteps}: Avg Reward (last 10) = {avg_reward:.2f}")
        
        return True
    
    def get_training_data(self) -> dict:
        """Return all logged data for saving."""
        return {
            "timesteps": self.timesteps_log,
            "rewards": self.rewards_log,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_episodes": len(self.episode_rewards)
        }


# ============================================================
# IPPO WRAPPER (Your original code - unchanged)
# ============================================================

def _get_agent_obs_helper(wrapper_instance, agent_id):
    """Helper function to get normalized observation for an agent."""
    raw_obs_volts_state = wrapper_instance.current_obs_dict[agent_id]
    normalized_raw = wrapper_instance._normalize_individual_obs(agent_id, raw_obs_volts_state)
    
    # Add Agent Type ID
    atype = wrapper_instance.ma_env.agent_types[agent_id]
    if atype == 'cap': 
        type_id = -1.0
    elif atype == 'reg': 
        type_id = 0.0
    elif atype == 'bat': 
        type_id = 1.0
    else:
        type_id = 0.0
    
    return np.array([normalized_raw[0], normalized_raw[1], type_id], dtype=np.float32)


class IPPO_Wrapper(gym.Env):
    """
    Wraps the Multi-Agent environment for Independent PPO (IPPO) training.
    All agents share the same policy network (parameter sharing).
    """
    
    def __init__(self, env_name='13Bus'):
        super().__init__()
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.agents = self.ma_env.agents
        
        # Define observation bounds for normalization
        self.original_ma_obs_bounds = {}
        for agent_id in self.agents:
            atype = self.ma_env.agent_types[agent_id]
            voltage_low, voltage_high = 0.8, 1.2

            if atype == 'cap':
                state_low, state_high = 0.0, 1.0
            elif atype == 'reg':
                state_low = 0.0
                state_high = float(self.ma_env.raw_env.reg_act_num - 1)
            elif atype == 'bat':
                state_low, state_high = 0.0, 1.0
            else:
                raise ValueError(f"Unknown agent type: {atype}")
            
            low = np.array([voltage_low, state_low], dtype=np.float32)
            high = np.array([voltage_high, state_high], dtype=np.float32)
            
            self.original_ma_obs_bounds[agent_id] = {
                'low': low,
                'high': high,
                'range': high - low
            }
        
        self.original_ma_act_spaces = self.ma_env.action_spaces
        
        # Unified spaces
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(33)
        
        self.current_obs_dict = None
        self.agent_idx = 0
        
    def _normalize_individual_obs(self, agent_id, obs):
        bounds = self.original_ma_obs_bounds[agent_id]
        low, high = bounds['low'], bounds['high']
        obs_range = bounds['range'] + 1e-8
        normalized_obs = (obs - low) / obs_range * 2 - 1
        return normalized_obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.ma_env.raw_env.seed(seed)
        self.current_obs_dict = self.ma_env.reset()
        self.agent_idx = 0
        return _get_agent_obs_helper(self, self.agents[0]), {}

    def step(self, action):
        current_agent = self.agents[self.agent_idx]
        
        if not hasattr(self, 'action_buffer'):
            self.action_buffer = {}
            
        self.action_buffer[current_agent] = action
        self.agent_idx += 1
        
        if self.agent_idx < len(self.agents):
            next_agent = self.agents[self.agent_idx]
            obs = _get_agent_obs_helper(self, next_agent)
            return obs, 0.0, False, False, {}
            
        else:
            # All agents acted - step the environment
            mapped_actions = {}
            for agent_id, raw_action in self.action_buffer.items():
                atype = self.ma_env.agent_types[agent_id]
                if atype == 'cap':
                    mapped_actions[agent_id] = raw_action % 2
                elif atype == 'reg':
                    max_reg = self.ma_env.raw_env.reg_act_num - 1
                    mapped_actions[agent_id] = min(raw_action, max_reg)
                elif atype == 'bat':
                    if self.ma_env.raw_env.bat_act_num == np.inf:
                        mapped_actions[agent_id] = np.array([(raw_action / 32.0) * 2 - 1], dtype=np.float32)
                    else:
                        max_bat = self.ma_env.raw_env.bat_act_num - 1
                        mapped_actions[agent_id] = min(raw_action, max_bat)
            
            obs_dict, reward_dict, done, info = self.ma_env.step(mapped_actions)
            
            self.action_buffer = {}
            self.agent_idx = 0
            self.current_obs_dict = obs_dict
            
            avg_reward = np.mean(list(reward_dict.values()))
            obs = _get_agent_obs_helper(self, self.agents[0])
            
            return obs, avg_reward, done, False, info


# ============================================================
# TRAINING FUNCTION (MODIFIED)
# ============================================================

def train_ippo(env_name='13Bus', steps=10000, seed=42, model_name=None, 
               save_training_data=True, verbose=1):
    """
    Train IPPO agent with seed control and logging.
    
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
    print(f"Training IPPO (Specialist Ensemble)")
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
    
    # Auto-detect best device
    # if torch.backends.mps.is_available():
    #     device = "mps"
    # elif torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    device = "cpu"
    
    # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # ===== CREATE ENVIRONMENT =====
    env = IPPO_Wrapper(env_name)
    
    # Create log directory
    log_dir = f"logs/specialist_{env_name}_seed{seed}/"
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
        seed=seed,  # Pass seed to PPO
        verbose=verbose,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    
    # ===== CREATE CALLBACK FOR LOGGING =====
    callback = TrainingMetricsCallback(log_freq=500, verbose=0)
    
    # ===== TRAIN =====
    start_time = datetime.now()
    model.learn(total_timesteps=steps, callback=callback)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # ===== SAVE MODEL =====
    if model_name is None:
        model_name = f"specialist_ensemble_{env_name}_seed{seed}"
    
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
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for k, v in training_data.items():
                if isinstance(v, np.ndarray):
                    serializable_data[k] = v.tolist()
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.floating):
                    serializable_data[k] = [float(x) for x in v]
                else:
                    serializable_data[k] = v
            json.dump(serializable_data, f, indent=2)
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
    parser = argparse.ArgumentParser(description='Train IPPO (Specialist Ensemble) Agent')
    
    parser.add_argument('--env_name', type=str, default='13Bus',
                       choices=['13Bus', '34Bus', '123Bus', '8500Node'],
                       help='Environment name')
    parser.add_argument('--steps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Custom model name (optional)')
    parser.add_argument('--no_save_data', action='store_true',
                       help='Disable saving training data')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0=silent, 1=progress)')
    
    args = parser.parse_args()
    
    train_ippo(
        env_name=args.env_name,
        steps=args.steps,
        seed=args.seed,
        model_name=args.model_name,
        save_training_data=not args.no_save_data,
        verbose=args.verbose
    )