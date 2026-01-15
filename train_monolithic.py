
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import powergym.env_register as env_register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd

class PowerGymWrapper(gym.Env):
    """
    Wrapper to convert PowerGym (old gym) to Gymnasium (for SB3)
    and normalize observations.
    """
    def __init__(self, env_name='13Bus'):
        self.env = env_register.make_env(env_name, wrap_observation=True)
        
        # Store original observation space bounds for normalization
        self.original_obs_low = self.env.observation_space.low
        self.original_obs_high = self.env.observation_space.high
        self.original_obs_range = self.original_obs_high - self.original_obs_low

        # Define normalized observation space [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=self.env.observation_space.shape, 
                                            dtype=np.float32)
        
        # Convert action space (no normalization needed for discrete actions)
        self.action_space = self._convert_space(self.env.action_space)
        
    def _convert_space(self, space):
        if isinstance(space, (gym.spaces.Box, spaces.Box)):
            return spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
        elif "MultiDiscrete" in str(type(space)):
             return spaces.MultiDiscrete(space.nvec)
        return space

    def _normalize_observation(self, obs):
        # Normalize obs to [-1, 1] range
        # (obs - low) / (high - low) * 2 - 1
        normalized_obs = (obs - self.original_obs_low) / (self.original_obs_range + 1e-8) * 2 - 1
        return normalized_obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        normalized_obs = self._normalize_observation(obs)
        return normalized_obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        normalized_obs = self._normalize_observation(obs)
        terminated = done
        truncated = False 
        return normalized_obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()

def train_monolithic(env_name='13Bus', steps=3000):
    log_dir = f"logs/monolithic_{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = PowerGymWrapper(env_name)
    env = Monitor(env, log_dir + "monitor.csv")
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    print(f"Training Monolithic Agent on {env_name} for {steps} steps...")
    model.learn(total_timesteps=steps)
    model.save(f"monolithic_agent_{env_name}")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Monolithic Agent')
    parser.add_argument('--env_name', type=str, default='13Bus', help='Environment name (e.g., 13Bus, 34Bus)')
    parser.add_argument('--steps', type=int, default=3000, help='Number of training steps')
    args = parser.parse_args()
    
    train_monolithic(env_name=args.env_name, steps=args.steps)
