"""
train_maddpg.py - Multi-Agent DDPG Baseline
=============================================
CTDE actor-critic with per-agent critics that observe the joint state+actions.
Uses Gumbel-Softmax for discrete action spaces (caps, regs, bats).

This is the most commonly used MARL baseline in the VVC literature
(Gao et al. TSG 2021, Wang et al. NeurIPS 2021 MAPDN benchmark).

Usage:
    python train_maddpg.py --env_name 13Bus --steps 50000 --seed 42
"""

import gymnasium as gym
import numpy as np
import argparse
import os
import json
import random
import copy
import time
from datetime import datetime
from collections import deque

from powergym.ma_env import MultiAgentPowerGrid

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ImportError:
    print("Error: torch not installed.")
    exit(1)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, next_state, done):
        self.buffer.append((state, actions, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), list(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# ACTOR AND CRITIC NETWORKS
# ============================================================

class Actor(nn.Module):
    """Per-agent actor: local_obs -> action logits."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs, explore=True, temperature=1.0):
        logits = self.forward(obs)
        if explore:
            action = F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            action = F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        return action


class Critic(nn.Module):
    """Centralized critic: [global_state, all_actions] -> Q-value."""
    def __init__(self, global_obs_dim, total_act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim + total_act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, global_obs, all_actions):
        x = torch.cat([global_obs, all_actions], dim=-1)
        return self.net(x)


# ============================================================
# MADDPG AGENT
# ============================================================

class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, act_dim, global_obs_dim,
                 total_act_dim, lr=1e-3, gamma=0.99, tau=0.01):
        self.agent_id = agent_id
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(global_obs_dim, total_act_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr * 2)

    def soft_update(self):
        for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)


# ============================================================
# MADDPG ENVIRONMENT WRAPPER
# ============================================================

class MADDPGEnvWrapper:
    """Thin wrapper to interface MultiAgentPowerGrid with MADDPG."""

    def __init__(self, env_name='13Bus'):
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.raw_env = self.ma_env.raw_env
        self.agents = self.ma_env.agents
        self.n_agents = len(self.agents)

        # Per-agent observation and action dimensions
        self.obs_dims = {}
        self.act_dims = {}
        for agent_id in self.agents:
            self.obs_dims[agent_id] = 3  # [voltage, state, type_id]
            atype = self.ma_env.agent_types[agent_id]
            if atype == 'cap':
                self.act_dims[agent_id] = 2
            elif atype == 'reg':
                self.act_dims[agent_id] = self.raw_env.reg_act_num
            elif atype == 'bat':
                if self.raw_env.bat_act_num == np.inf:
                    self.act_dims[agent_id] = 33  # Discretize continuous
                else:
                    self.act_dims[agent_id] = self.raw_env.bat_act_num

        # Uniform obs/act dims for simplicity (pad to max)
        self.max_act_dim = max(self.act_dims.values())
        self.obs_dim = 3
        self.total_act_dim = self.max_act_dim * self.n_agents

        # Global observation: all bus voltages (averaged per bus)
        self.n_buses = len(self.raw_env.all_bus_names)
        self.global_obs_dim = self.n_buses

        self.current_obs_dict = None

    def _get_local_obs(self, agent_id):
        raw = self.current_obs_dict[agent_id]
        # Normalize voltage around 1.0, state around 0.5
        v_norm = (raw[0] - 1.0) * 10.0
        s_norm = raw[1] / 16.0 - 1.0  # rough normalization
        atype = self.ma_env.agent_types[agent_id]
        type_id = {'cap': -1.0, 'reg': 0.0, 'bat': 1.0}.get(atype, 0.0)
        return np.array([v_norm, s_norm, type_id], dtype=np.float32)

    def _get_global_obs(self):
        bus_volts = []
        for bus_name in self.raw_env.all_bus_names:
            v_data = self.raw_env.circuit.bus_voltage(bus_name)
            v_mags = [v_data[i] for i in range(len(v_data)) if i % 2 == 0]
            bus_volts.append((np.mean(v_mags) - 1.0) * 10.0 if v_mags else 0.0)
        return np.array(bus_volts, dtype=np.float32)

    def reset(self):
        self.current_obs_dict = self.ma_env.reset()
        local_obs = [self._get_local_obs(aid) for aid in self.agents]
        global_obs = self._get_global_obs()
        return local_obs, global_obs

    def step(self, action_indices):
        """
        Args:
            action_indices: list of integer actions per agent
        """
        mapped = {}
        for i, agent_id in enumerate(self.agents):
            atype = self.ma_env.agent_types[agent_id]
            act = action_indices[i]
            if atype == 'cap':
                mapped[agent_id] = act % 2
            elif atype == 'reg':
                mapped[agent_id] = min(act, self.raw_env.reg_act_num - 1)
            elif atype == 'bat':
                if self.raw_env.bat_act_num == np.inf:
                    mapped[agent_id] = np.array(
                        [(act / 32.0) * 2 - 1], dtype=np.float32)
                else:
                    mapped[agent_id] = min(act, self.raw_env.bat_act_num - 1)

        obs_dict, reward_dict, done, info = self.ma_env.step(mapped)
        self.current_obs_dict = obs_dict

        local_obs = [self._get_local_obs(aid) for aid in self.agents]
        global_obs = self._get_global_obs()
        avg_reward = np.mean(list(reward_dict.values()))

        return local_obs, global_obs, avg_reward, done, info


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_maddpg(env_name='13Bus', steps=50000, seed=42, model_name=None,
                 save_training_data=True, verbose=1,
                 batch_size=256, buffer_size=100000, lr=1e-3,
                 gamma=0.99, tau=0.01, update_every=100,
                 warmup_steps=2000, temperature_start=2.0,
                 temperature_end=0.5):
    """Train MADDPG agents."""

    print(f"\n{'='*60}")
    print(f"Training MADDPG (Centralized Critics)")
    print(f"Environment: {env_name}")
    print(f"Timesteps: {steps}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create environment
    env = MADDPGEnvWrapper(env_name)

    # Create agents
    maddpg_agents = []
    for i, agent_id in enumerate(env.agents):
        agent = MADDPGAgent(
            agent_id=agent_id,
            obs_dim=env.obs_dim,
            act_dim=env.act_dims[agent_id],
            global_obs_dim=env.global_obs_dim,
            total_act_dim=sum(env.act_dims[aid] for aid in env.agents),
            lr=lr, gamma=gamma, tau=tau
        )
        maddpg_agents.append(agent)

    buffer = ReplayBuffer(buffer_size)

    # Logging
    episode_rewards_log = []
    timesteps_log = []
    rewards_log = []

    total_steps = 0
    episode_count = 0
    start_time = datetime.now()

    while total_steps < steps:
        local_obs, global_obs = env.reset()
        ep_reward = 0
        done = False

        while not done and total_steps < steps:
            # Anneal temperature
            progress = min(1.0, total_steps / steps)
            temperature = temperature_start + (temperature_end - temperature_start) * progress

            # Select actions
            actions_onehot = []
            action_indices = []
            explore = total_steps < warmup_steps or random.random() < max(0.05, 1.0 - progress)

            for i, agent in enumerate(maddpg_agents):
                obs_t = torch.FloatTensor(local_obs[i]).unsqueeze(0)
                if total_steps < warmup_steps:
                    # Random exploration during warmup
                    act_idx = random.randint(0, agent.act_dim - 1)
                    act_oh = np.zeros(agent.act_dim)
                    act_oh[act_idx] = 1.0
                else:
                    with torch.no_grad():
                        act_oh_t = agent.actor.get_action(
                            obs_t, explore=explore, temperature=temperature)
                        act_oh = act_oh_t.squeeze(0).numpy()
                    act_idx = int(np.argmax(act_oh))
                actions_onehot.append(act_oh)
                action_indices.append(act_idx)

            # Step environment
            next_local_obs, next_global_obs, reward, done, info = env.step(action_indices)
            ep_reward += reward
            total_steps += 1

            # Store transition
            buffer.push(
                global_obs,
                actions_onehot,
                reward,
                next_global_obs,
                float(done)
            )

            local_obs = next_local_obs
            global_obs = next_global_obs

            # Update networks
            if len(buffer) >= batch_size and total_steps % update_every == 0:
                _update_agents(maddpg_agents, env, buffer, batch_size, local_obs)

        episode_count += 1
        episode_rewards_log.append(ep_reward)

        if episode_count % 5 == 0:
            timesteps_log.append(total_steps)
            recent = episode_rewards_log[-10:] if len(episode_rewards_log) >= 10 else episode_rewards_log
            rewards_log.append(float(np.mean(recent)))
            if verbose > 0:
                print(f"  Episode {episode_count} | Steps: {total_steps} | "
                      f"Reward: {ep_reward:.2f} | Avg10: {np.mean(recent):.2f}")

    training_time = (datetime.now() - start_time).total_seconds()

    # Save models
    if model_name is None:
        model_name = f"maddpg_{env_name}_seed{seed}"

    os.makedirs(os.path.dirname(model_name) if os.path.dirname(model_name) else '.', exist_ok=True)
    save_dict = {
        f"actor_{i}": agent.actor.state_dict()
        for i, agent in enumerate(maddpg_agents)
    }
    torch.save(save_dict, f"{model_name}.pt")
    print(f"\nModels saved: {model_name}.pt")

    training_data = {
        "timesteps": timesteps_log,
        "rewards": rewards_log,
        "episode_rewards": [float(r) for r in episode_rewards_log],
        "episode_lengths": [],
        "total_episodes": episode_count,
        "seed": seed,
        "env_name": env_name,
        "total_timesteps": steps,
        "training_time_seconds": training_time
    }

    if save_training_data:
        data_path = f"{model_name}_training_data.json"
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2)

    final_reward = np.mean(episode_rewards_log[-10:]) if episode_rewards_log else 0
    print(f"\nTraining Complete! Final Avg Reward: {final_reward:.2f}")
    print(f"Training Time: {training_time:.1f}s\n")

    return f"{model_name}.pt", training_data


def _update_agents(maddpg_agents, env, buffer, batch_size, current_local_obs):
    """Perform one round of MADDPG updates for all agents."""
    states, actions_list, rewards, next_states, dones = buffer.sample(batch_size)

    states_t = torch.FloatTensor(states)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(-1)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones).unsqueeze(-1)

    # Concatenate all actions
    all_actions_t = torch.cat(
        [torch.FloatTensor(np.array([a[i] for a in actions_list]))
         for i in range(len(maddpg_agents))], dim=-1)

    # Target actions for next state
    with torch.no_grad():
        target_actions = []
        for i, agent in enumerate(maddpg_agents):
            # Use a dummy local obs (global state is what matters for critic)
            obs_t = torch.FloatTensor(
                np.array([current_local_obs[i]] * batch_size))
            target_act = agent.target_actor.get_action(obs_t, explore=False)
            target_actions.append(target_act)
        target_all_actions = torch.cat(target_actions, dim=-1)

    for i, agent in enumerate(maddpg_agents):
        # --- Update Critic ---
        with torch.no_grad():
            target_q = agent.target_critic(next_states_t, target_all_actions)
            target_value = rewards_t + agent.gamma * (1 - dones_t) * target_q

        current_q = agent.critic(states_t, all_actions_t)
        critic_loss = F.mse_loss(current_q, target_value)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # --- Update Actor ---
        obs_t = torch.FloatTensor(
            np.array([current_local_obs[i]] * batch_size))
        new_act = agent.actor.get_action(obs_t, explore=True, temperature=1.0)

        # Replace this agent's actions in the joint action vector
        new_all_actions = all_actions_t.clone()
        act_start = sum(maddpg_agents[j].act_dim for j in range(i))
        act_end = act_start + agent.act_dim
        new_all_actions[:, act_start:act_end] = new_act

        actor_loss = -agent.critic(states_t, new_all_actions).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        # --- Soft update targets ---
        agent.soft_update()


# ============================================================
# EVALUATION FUNCTION (for run_exp.py integration)
# ============================================================

def evaluate_maddpg(env_name, model_path, n_episodes=10, seed=None):
    """Evaluate trained MADDPG agents."""
    env = MADDPGEnvWrapper(env_name)

    # Load actor networks
    checkpoint = torch.load(model_path, map_location='cpu')
    actors = []
    for i, agent_id in enumerate(env.agents):
        act_dim = env.act_dims[agent_id]
        actor = Actor(env.obs_dim, act_dim)
        actor.load_state_dict(checkpoint[f"actor_{i}"])
        actor.eval()
        actors.append(actor)

    episode_rewards = []
    episode_violations = []
    episode_losses = []

    total_steps = 0
    t_start = time.time()  # same wall-clock-per-step methodology as run_exp.py

    for ep in range(n_episodes):
        local_obs, global_obs = env.reset()
        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []

        while not done:
            action_indices = []
            for i, actor in enumerate(actors):
                obs_t = torch.FloatTensor(local_obs[i]).unsqueeze(0)
                with torch.no_grad():
                    act_oh = actor.get_action(obs_t, explore=False)
                action_indices.append(int(act_oh.argmax(-1).item()))

            local_obs, global_obs, reward, done, info = env.step(action_indices)
            ep_reward += reward
            total_steps += 1

            if isinstance(info, dict):
                if 'constraint_cost' in info:
                    ep_violations.append(info['constraint_cost'])
                elif 'vol_reward' in info:
                    ep_violations.append(abs(info['vol_reward']))
                if 'power_loss_ratio' in info:
                    ep_losses.append(info['power_loss_ratio'])

        episode_rewards.append(ep_reward)
        if ep_violations:
            episode_violations.append(np.mean(ep_violations))
        if ep_losses:
            episode_losses.append(np.mean(ep_losses))

    inference_time = time.time() - t_start

    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_all": episode_rewards,
        "violation_mean": np.mean(episode_violations) if episode_violations else 0,
        "violation_std": np.std(episode_violations) if episode_violations else 0,
        "loss_mean": np.mean(episode_losses) if episode_losses else 0,
        "loss_std": np.std(episode_losses) if episode_losses else 0,
        "inference_time_total": inference_time,
        "inference_time_per_step": inference_time / max(total_steps, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MADDPG Agents')
    parser.add_argument('--env_name', type=str, default='13Bus',
                        choices=['13Bus', '34Bus', '123Bus', '8500Node'])
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    train_maddpg(
        env_name=args.env_name, steps=args.steps, seed=args.seed,
        model_name=args.model_name, verbose=args.verbose
    )