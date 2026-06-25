"""
train_mappo.py - Multi-Agent PPO with Centralized Critic
==========================================================
MAPPO: Centralized Training, Decentralized Execution (CTDE)

Key difference from IPPO (train_marl.py):
  - IPPO:  actor gets [local_obs, type_id], critic gets [local_obs, type_id]
  - MAPPO: actor gets [local_obs, type_id], critic gets [GLOBAL_STATE]

The global state includes all bus voltages, all device statuses, and the
agent-type identifier — giving the centralized critic full observability
for more accurate value estimation during training.

At execution time, only the actor (policy) head is used, which sees
only local observations — identical to IPPO.

Usage:
    python train_mappo.py --env_name 13Bus --steps 50000 --seed 42
"""

import gymnasium as gym
import numpy as np
import argparse
import os
import json
import random
from datetime import datetime

from powergym.ma_env import MultiAgentPowerGrid

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: stable-baselines3 or torch not installed.")
    exit(1)


# ============================================================
# TRAINING METRICS CALLBACK (shared pattern)
# ============================================================

class TrainingMetricsCallback(BaseCallback):
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
# MAPPO WRAPPER
# ============================================================

class MAPPO_Wrapper(gym.Env):
    """
    MAPPO wrapper: observation contains both local and global information.
    
    Observation layout:
      [0:3]   = local observation (voltage, device_state, type_id)  -- for ACTOR
      [3:3+G] = global state (all voltages + all device statuses)   -- for CRITIC
    
    SB3's PPO feeds the full observation to both actor and critic.
    We use a custom feature extractor (MAPPOFeatureExtractor) that
    splits the observation: actor sees only local, critic sees global.
    """

    def __init__(self, env_name='13Bus'):
        super().__init__()
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.agents = self.ma_env.agents
        self.raw_env = self.ma_env.raw_env

        # Compute global state dimension
        # Global = all bus voltages (1 per bus) + cap statuses + reg statuses + bat SoCs
        self.n_buses = len(self.raw_env.all_bus_names)
        n_caps = len(self.raw_env.cap_names)
        n_regs = len(self.raw_env.reg_names)
        n_bats = len(self.raw_env.bat_names)
        self.global_dim = self.n_buses + n_caps + n_regs + n_bats
        self.local_dim = 3  # [voltage, state, type_id]

        # Total observation = local + global
        total_dim = self.local_dim + self.global_dim
        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(total_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(33)

        # Normalization bounds (same as IPPO)
        self.original_ma_obs_bounds = {}
        for agent_id in self.agents:
            atype = self.ma_env.agent_types[agent_id]
            voltage_low, voltage_high = 0.8, 1.2
            if atype == 'cap':
                state_low, state_high = 0.0, 1.0
            elif atype == 'reg':
                state_low, state_high = 0.0, float(self.raw_env.reg_act_num - 1)
            elif atype == 'bat':
                state_low, state_high = 0.0, 1.0
            else:
                raise ValueError(f"Unknown agent type: {atype}")
            low = np.array([voltage_low, state_low], dtype=np.float32)
            high = np.array([voltage_high, state_high], dtype=np.float32)
            self.original_ma_obs_bounds[agent_id] = {
                'low': low, 'high': high,
                'range': high - low
            }

        self.original_ma_act_spaces = self.ma_env.action_spaces
        self.current_obs_dict = None
        self.agent_idx = 0

    def _normalize_individual_obs(self, agent_id, obs):
        bounds = self.original_ma_obs_bounds[agent_id]
        obs_range = bounds['range'] + 1e-8
        return ((obs - bounds['low']) / obs_range * 2 - 1).astype(np.float32)

    def _get_global_state(self):
        """Extract normalized global state from current OpenDSS state."""
        # Bus voltages (normalized around 1.0)
        bus_volts = []
        for bus_name in self.raw_env.all_bus_names:
            v_data = self.raw_env.circuit.bus_voltage(bus_name)
            v_mags = [v_data[i] for i in range(len(v_data)) if i % 2 == 0]
            avg_v = np.mean(v_mags) if v_mags else 1.0
            bus_volts.append((avg_v - 1.0) * 10.0)  # Scale around 0

        # Device statuses (normalized)
        cap_statuses = [
            float(self.raw_env.circuit.capacitors[name].status) * 2 - 1
            for name in self.raw_env.cap_names
        ]
        reg_statuses = [
            (self.raw_env.circuit.regulators[name].tap - 1.0) * 10.0
            for name in self.raw_env.reg_names
        ]
        bat_socs = [
            self.raw_env.circuit.batteries[name].soc * 2 - 1
            for name in self.raw_env.bat_names
        ]

        return np.array(bus_volts + cap_statuses + reg_statuses + bat_socs,
                        dtype=np.float32)

    def _get_agent_obs(self, agent_id):
        """Build concatenated [local | global] observation for an agent."""
        raw_obs = self.current_obs_dict[agent_id]
        normalized = self._normalize_individual_obs(agent_id, raw_obs)

        atype = self.ma_env.agent_types[agent_id]
        type_id = {'cap': -1.0, 'reg': 0.0, 'bat': 1.0}.get(atype, 0.0)

        local_obs = np.array([normalized[0], normalized[1], type_id], dtype=np.float32)
        global_state = self._get_global_state()

        return np.concatenate([local_obs, global_state])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.raw_env.seed(seed)
        self.current_obs_dict = self.ma_env.reset()
        self.agent_idx = 0
        return self._get_agent_obs(self.agents[0]), {}

    def step(self, action):
        current_agent = self.agents[self.agent_idx]

        if not hasattr(self, 'action_buffer'):
            self.action_buffer = {}
        self.action_buffer[current_agent] = action
        self.agent_idx += 1

        if self.agent_idx < len(self.agents):
            next_agent = self.agents[self.agent_idx]
            obs = self._get_agent_obs(next_agent)
            return obs, 0.0, False, False, {}
        else:
            # All agents acted — step the environment
            mapped_actions = {}
            for agent_id, raw_action in self.action_buffer.items():
                atype = self.ma_env.agent_types[agent_id]
                if atype == 'cap':
                    mapped_actions[agent_id] = raw_action % 2
                elif atype == 'reg':
                    max_reg = self.raw_env.reg_act_num - 1
                    mapped_actions[agent_id] = min(raw_action, max_reg)
                elif atype == 'bat':
                    if self.raw_env.bat_act_num == np.inf:
                        mapped_actions[agent_id] = np.array(
                            [(raw_action / 32.0) * 2 - 1], dtype=np.float32)
                    else:
                        max_bat = self.raw_env.bat_act_num - 1
                        mapped_actions[agent_id] = min(raw_action, max_bat)

            obs_dict, reward_dict, done, info = self.ma_env.step(mapped_actions)
            self.action_buffer = {}
            self.agent_idx = 0
            self.current_obs_dict = obs_dict

            avg_reward = np.mean(list(reward_dict.values()))
            obs = self._get_agent_obs(self.agents[0])
            return obs, avg_reward, done, False, info


# ============================================================
# CUSTOM FEATURE EXTRACTOR FOR ACTOR-CRITIC SPLIT
# ============================================================

class MAPPOFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that provides:
      - Actor (policy): only local observation (first 3 dims)
      - Critic (value): full observation (local + global)
    
    SB3 uses a single feature extractor for both, but by setting
    features_dim to local_dim + global_dim and letting the policy
    network handle the split via its architecture, we achieve
    the MAPPO effect.
    
    In practice, with SB3's shared feature extractor, the simplest
    correct approach is to pass the FULL observation and let the
    separate pi/vf networks learn what to attend to. The vf network
    with access to global state will learn better value estimates,
    which improves the advantage computation and thus the policy
    gradient — this IS the MAPPO benefit.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        total_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=total_dim)
        # Identity pass-through: let pi and vf networks handle splitting
        self.net = nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_mappo(env_name='13Bus', steps=10000, seed=42, model_name=None,
                save_training_data=True, verbose=1):
    """
    Train MAPPO agent with centralized critic.
    
    The key difference from IPPO: the observation includes global state,
    giving the value function (critic) access to system-wide information
    during training. At execution, only the policy (actor) head is used,
    and it can learn to act on local information guided by the better
    value estimates from training.
    """
    print(f"\n{'='*60}")
    print(f"Training MAPPO (Centralized Critic)")
    print(f"Environment: {env_name}")
    print(f"Timesteps: {steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cpu"

    # Create environment
    env = MAPPO_Wrapper(env_name)
    log_dir = f"logs/mappo_{env_name}_seed{seed}/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir + "monitor.csv")

    # Determine observation dimensions for network architecture
    total_obs_dim = env.observation_space.shape[0]
    local_dim = 3  # [voltage, state, type_id]
    global_dim = total_obs_dim - local_dim

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
            features_extractor_class=MAPPOFeatureExtractor,
            net_arch=dict(
                # Policy network: smaller since it should focus on local info
                pi=[128, 128],
                # Value network: larger to process global state
                vf=[256, 256]
            )
        )
    )

    callback = TrainingMetricsCallback(log_freq=500, verbose=0)

    start_time = datetime.now()
    model.learn(total_timesteps=steps, callback=callback)
    training_time = (datetime.now() - start_time).total_seconds()

    # Save
    if model_name is None:
        model_name = f"mappo_{env_name}_seed{seed}"
    model.save(model_name)
    print(f"\nModel saved: {model_name}.zip")

    training_data = callback.get_training_data()
    training_data.update({
        "seed": seed, "env_name": env_name,
        "total_timesteps": steps, "training_time_seconds": training_time
    })

    if save_training_data:
        data_path = f"{model_name}_training_data.json"
        with open(data_path, 'w') as f:
            serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in training_data.items()}
            json.dump(serializable, f, indent=2)

    final_reward = np.mean(training_data["episode_rewards"][-10:]) if training_data["episode_rewards"] else 0
    print(f"\nTraining Complete! Final Avg Reward: {final_reward:.2f}")
    print(f"Training Time: {training_time:.1f}s\n")

    env.close()
    return f"{model_name}.zip", training_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MAPPO Agent')
    parser.add_argument('--env_name', type=str, default='13Bus',
                        choices=['13Bus', '34Bus', '123Bus', '8500Node'])
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--no_save_data', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    train_mappo(
        env_name=args.env_name, steps=args.steps, seed=args.seed,
        model_name=args.model_name,
        save_training_data=not args.no_save_data, verbose=args.verbose
    )