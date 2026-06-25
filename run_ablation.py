"""
run_ablation.py - Ablation Study Runner
=========================================
Systematically evaluates each component of the Specialist Ensemble:
  1. Full model (parameter sharing + type embedding + local obs)
  2. No parameter sharing (separate policy per device type)
  3. No type embedding (type_id zeroed out)
  4. No sharing + no embedding (fully independent agents)
  5. Global observations (instead of local)

Run on a single system (34Bus recommended - medium complexity, shows
clearest differences) with multiple seeds.

Usage:
    python run_ablation.py --env_name 34Bus --seeds 5 --steps 100000
    python run_ablation.py --env_name 13Bus --seeds 5 --steps 50000
"""

import gymnasium as gym
import numpy as np
import argparse
import os
import json
import random
from datetime import datetime

from powergym.ma_env import MultiAgentPowerGrid

# Reuse the paper's robust statistics (IQM + bootstrap 95% CI) so the ablation
# table is consistent with the main results in run_exp.py.
from enhanced_statistics import compute_statistics

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
except ImportError:
    print("Error: stable-baselines3 or torch not installed.")
    exit(1)


SEEDS = [42, 123, 456, 789, 1011]


# ============================================================
# TRAINING METRICS CALLBACK (shared)
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.timesteps_log = []
        self.rewards_log = []
        self._current_ep_reward = 0

    def _on_step(self) -> bool:
        self._current_ep_reward += self.locals.get('rewards', [0])[0]
        if any(self.locals.get('dones', [False])):
            self.episode_rewards.append(self._current_ep_reward)
            self._current_ep_reward = 0
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            recent = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            self.timesteps_log.append(self.num_timesteps)
            self.rewards_log.append(np.mean(recent))
        return True

    def get_training_data(self):
        return {
            "timesteps": self.timesteps_log,
            "rewards": self.rewards_log,
            "episode_rewards": self.episode_rewards,
            "total_episodes": len(self.episode_rewards)
        }


# ============================================================
# CONFIGURABLE IPPO WRAPPER (supports ablation flags)
# ============================================================

class AblationIPPOWrapper(gym.Env):
    """
    IPPO wrapper with ablation flags to disable specific components.
    
    Args:
        env_name: PowerGym environment name
        use_type_embed: If False, agent type_id is zeroed out
        use_global_obs: If True, observation includes all bus voltages
    """

    def __init__(self, env_name='13Bus', use_type_embed=True,
                 use_global_obs=False):
        super().__init__()
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.agents = self.ma_env.agents
        self.raw_env = self.ma_env.raw_env
        self.use_type_embed = use_type_embed
        self.use_global_obs = use_global_obs

        # Obs bounds for normalization
        self.original_ma_obs_bounds = {}
        for agent_id in self.agents:
            atype = self.ma_env.agent_types[agent_id]
            v_low, v_high = 0.8, 1.2
            if atype == 'cap':
                s_low, s_high = 0.0, 1.0
            elif atype == 'reg':
                s_low, s_high = 0.0, float(self.raw_env.reg_act_num - 1)
            elif atype == 'bat':
                s_low, s_high = 0.0, 1.0
            else:
                raise ValueError(f"Unknown type: {atype}")
            low = np.array([v_low, s_low], dtype=np.float32)
            high = np.array([v_high, s_high], dtype=np.float32)
            self.original_ma_obs_bounds[agent_id] = {
                'low': low, 'high': high, 'range': high - low
            }

        # Observation dimension
        self.local_dim = 3  # [voltage, state, type_id]
        if use_global_obs:
            self.n_buses = len(self.raw_env.all_bus_names)
            obs_dim = self.local_dim + self.n_buses
        else:
            obs_dim = self.local_dim

        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(33)

        self.current_obs_dict = None
        self.agent_idx = 0

    def _normalize(self, agent_id, obs):
        b = self.original_ma_obs_bounds[agent_id]
        return ((obs - b['low']) / (b['range'] + 1e-8) * 2 - 1).astype(np.float32)

    def _get_agent_obs(self, agent_id):
        raw = self.current_obs_dict[agent_id]
        norm = self._normalize(agent_id, raw)

        if self.use_type_embed:
            atype = self.ma_env.agent_types[agent_id]
            type_id = {'cap': -1.0, 'reg': 0.0, 'bat': 1.0}.get(atype, 0.0)
        else:
            type_id = 0.0  # Ablation: no type information

        local = np.array([norm[0], norm[1], type_id], dtype=np.float32)

        if self.use_global_obs:
            bus_volts = []
            for bus in self.raw_env.all_bus_names:
                v = self.raw_env.circuit.bus_voltage(bus)
                mags = [v[i] for i in range(len(v)) if i % 2 == 0]
                bus_volts.append((np.mean(mags) - 1.0) * 10.0 if mags else 0.0)
            return np.concatenate([local, np.array(bus_volts, dtype=np.float32)])
        else:
            return local

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
            return self._get_agent_obs(self.agents[self.agent_idx]), 0.0, False, False, {}
        else:
            mapped = {}
            for aid, raw_act in self.action_buffer.items():
                atype = self.ma_env.agent_types[aid]
                if atype == 'cap':
                    mapped[aid] = raw_act % 2
                elif atype == 'reg':
                    mapped[aid] = min(raw_act, self.raw_env.reg_act_num - 1)
                elif atype == 'bat':
                    if self.raw_env.bat_act_num == np.inf:
                        mapped[aid] = np.array([(raw_act / 32.0) * 2 - 1], dtype=np.float32)
                    else:
                        mapped[aid] = min(raw_act, self.raw_env.bat_act_num - 1)

            obs_dict, reward_dict, done, info = self.ma_env.step(mapped)
            self.action_buffer = {}
            self.agent_idx = 0
            self.current_obs_dict = obs_dict
            avg_reward = np.mean(list(reward_dict.values()))
            return self._get_agent_obs(self.agents[0]), avg_reward, done, False, info


# ============================================================
# TRAINING WITH SEPARATE POLICIES (No Parameter Sharing)
# ============================================================

def train_no_sharing(env_name, steps, seed, model_name, verbose=0):
    """
    Train separate PPO policies for each device TYPE (not each device).
    This is the 'no parameter sharing' ablation.
    
    Instead of 1 shared policy for all agents, we train 3 separate
    policies: one for caps, one for regs, one for bats.
    """
    print(f"  Training No-Sharing variant (3 separate policies)...")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    ma_env = MultiAgentPowerGrid(env_name)
    agent_types = {aid: ma_env.agent_types[aid] for aid in ma_env.agents}

    # Group agents by type
    type_groups = {'cap': [], 'reg': [], 'bat': []}
    for aid, atype in agent_types.items():
        type_groups[atype].append(aid)

    # Train one policy per type using a filtered wrapper
    type_models = {}
    total_time = 0

    for dtype in ['cap', 'reg', 'bat']:
        if not type_groups[dtype]:
            continue

        # Create a wrapper that only cycles through this type's agents
        env = TypeFilteredWrapper(env_name, device_type=dtype)
        log_dir = f"logs/ablation_noshare_{dtype}_{env_name}_seed{seed}/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir + "monitor.csv")

        model = PPO(
            "MlpPolicy", env, device="cpu",
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, seed=seed, verbose=verbose,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        )

        start = datetime.now()
        # Each type-specific policy trains for the FULL per-policy budget,
        # matching the shared policy's budget (standard no-sharing IPPO trains
        # every policy on the full rollout). NOTE: total compute for this
        # variant is therefore ~3x `steps`, since 3 policies each run `steps`.
        model.learn(total_timesteps=steps)
        total_time += (datetime.now() - start).total_seconds()

        path = f"{model_name}_{dtype}"
        model.save(path)
        type_models[dtype] = path
        env.close()

    return type_models, total_time


class TypeFilteredWrapper(gym.Env):
    """Wrapper that only exposes agents of a specific device type."""

    def __init__(self, env_name, device_type='cap'):
        super().__init__()
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.raw_env = self.ma_env.raw_env
        self.device_type = device_type

        # Filter agents to this type only
        self.agents = [
            aid for aid in self.ma_env.agents
            if self.ma_env.agent_types[aid] == device_type
        ]
        self.all_agents = self.ma_env.agents

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        if device_type == 'cap':
            self.action_space = gym.spaces.Discrete(2)
        elif device_type == 'reg':
            self.action_space = gym.spaces.Discrete(self.raw_env.reg_act_num)
        else:
            self.action_space = gym.spaces.Discrete(
                33 if self.raw_env.bat_act_num != np.inf else 33)

        self.current_obs_dict = None
        self.agent_idx = 0

        # Bounds
        self.obs_bounds = {}
        for aid in self.agents:
            atype = self.ma_env.agent_types[aid]
            v_low, v_high = 0.8, 1.2
            if atype == 'cap':
                s_low, s_high = 0.0, 1.0
            elif atype == 'reg':
                s_low, s_high = 0.0, float(self.raw_env.reg_act_num - 1)
            else:
                s_low, s_high = 0.0, 1.0
            low = np.array([v_low, s_low], dtype=np.float32)
            high = np.array([v_high, s_high], dtype=np.float32)
            self.obs_bounds[aid] = {'low': low, 'high': high, 'range': high - low}

    def _get_obs(self, agent_id):
        raw = self.current_obs_dict[agent_id]
        b = self.obs_bounds[agent_id]
        norm = ((raw - b['low']) / (b['range'] + 1e-8) * 2 - 1).astype(np.float32)
        atype = self.ma_env.agent_types[agent_id]
        tid = {'cap': -1.0, 'reg': 0.0, 'bat': 1.0}[atype]
        return np.array([norm[0], norm[1], tid], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.raw_env.seed(seed)
        self.current_obs_dict = self.ma_env.reset()
        self.agent_idx = 0
        if not self.agents:
            return np.zeros(3, dtype=np.float32), {}
        return self._get_obs(self.agents[0]), {}

    def step(self, action):
        if not self.agents:
            return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

        current = self.agents[self.agent_idx]
        if not hasattr(self, 'action_buffer'):
            self.action_buffer = {}
        self.action_buffer[current] = action
        self.agent_idx += 1

        if self.agent_idx < len(self.agents):
            return self._get_obs(self.agents[self.agent_idx]), 0.0, False, False, {}
        else:
            # Build full action dict with defaults for non-controlled types
            full_actions = {}
            for aid in self.all_agents:
                atype = self.ma_env.agent_types[aid]
                if aid in self.action_buffer:
                    raw_act = self.action_buffer[aid]
                    if atype == 'cap':
                        full_actions[aid] = raw_act % 2
                    elif atype == 'reg':
                        full_actions[aid] = min(raw_act, self.raw_env.reg_act_num - 1)
                    elif atype == 'bat':
                        if self.raw_env.bat_act_num == np.inf:
                            full_actions[aid] = np.array(
                                [(raw_act / 32.0) * 2 - 1], dtype=np.float32)
                        else:
                            full_actions[aid] = min(raw_act, self.raw_env.bat_act_num - 1)
                else:
                    # Default actions for other device types
                    if atype == 'cap':
                        full_actions[aid] = 1  # ON
                    elif atype == 'reg':
                        full_actions[aid] = self.raw_env.reg_act_num // 2  # Mid tap
                    elif atype == 'bat':
                        if self.raw_env.bat_act_num == np.inf:
                            full_actions[aid] = np.array([0.0], dtype=np.float32)
                        else:
                            full_actions[aid] = self.raw_env.bat_act_num // 2  # Idle

            obs_dict, reward_dict, done, info = self.ma_env.step(full_actions)
            self.action_buffer = {}
            self.agent_idx = 0
            self.current_obs_dict = obs_dict

            # Reward only from this type's agents
            type_rewards = [reward_dict[aid] for aid in self.agents if aid in reward_dict]
            avg_reward = np.mean(type_rewards) if type_rewards else 0.0
            return self._get_obs(self.agents[0]), avg_reward, done, False, info


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

ABLATION_CONFIGS = {
    'full_model': {
        'description': 'Full Specialist Ensemble (sharing + type embed + local obs)',
        'use_type_embed': True,
        'use_global_obs': False,
        'use_sharing': True,
    },
    'no_type_embed': {
        'description': 'No agent-type embedding (type_id = 0 for all)',
        'use_type_embed': False,
        'use_global_obs': False,
        'use_sharing': True,
    },
    'no_sharing': {
        'description': 'No parameter sharing (separate policy per device type)',
        'use_type_embed': True,
        'use_global_obs': False,
        'use_sharing': False,
    },
    'global_obs': {
        'description': 'Global observations (local + all bus voltages)',
        'use_type_embed': True,
        'use_global_obs': True,
        'use_sharing': True,
    },
    'no_sharing_no_embed': {
        'description': 'Fully independent (no sharing + no type embed)',
        'use_type_embed': False,
        'use_global_obs': False,
        'use_sharing': False,
    },
}


# ============================================================
# NO-SHARING EVALUATION
# ============================================================

def evaluate_no_sharing(env_name, type_models, seed, n_episodes=10):
    """Evaluate the no-sharing variant.

    The three device-type policies are rolled out jointly on the full
    multi-agent environment: each agent's action is produced by the policy
    for its device type. Observation normalization and action mapping mirror
    TypeFilteredWrapper exactly so eval matches the training distribution.
    """
    ma_env = MultiAgentPowerGrid(env_name)
    raw_env = ma_env.raw_env
    agents = ma_env.agents
    agent_types = {aid: ma_env.agent_types[aid] for aid in agents}

    # Load the per-type policies that were actually trained
    models = {}
    for dtype, path in type_models.items():
        if os.path.exists(path + ".zip"):
            models[dtype] = PPO.load(path)

    # Obs bounds (mirror TypeFilteredWrapper)
    obs_bounds = {}
    for aid in agents:
        atype = agent_types[aid]
        v_low, v_high = 0.8, 1.2
        if atype == 'cap':
            s_low, s_high = 0.0, 1.0
        elif atype == 'reg':
            s_low, s_high = 0.0, float(raw_env.reg_act_num - 1)
        else:
            s_low, s_high = 0.0, 1.0
        low = np.array([v_low, s_low], dtype=np.float32)
        high = np.array([v_high, s_high], dtype=np.float32)
        obs_bounds[aid] = {'low': low, 'high': high, 'range': high - low}

    def get_obs(aid, obs_dict):
        raw = obs_dict[aid]
        b = obs_bounds[aid]
        norm = ((raw - b['low']) / (b['range'] + 1e-8) * 2 - 1).astype(np.float32)
        tid = {'cap': -1.0, 'reg': 0.0, 'bat': 1.0}[agent_types[aid]]
        return np.array([norm[0], norm[1], tid], dtype=np.float32)

    def map_action(aid, raw_act):
        atype = agent_types[aid]
        if atype == 'cap':
            return raw_act % 2
        elif atype == 'reg':
            return min(raw_act, raw_env.reg_act_num - 1)
        else:  # bat
            if raw_env.bat_act_num == np.inf:
                return np.array([(raw_act / 32.0) * 2 - 1], dtype=np.float32)
            return min(raw_act, raw_env.bat_act_num - 1)

    def default_action(aid):
        atype = agent_types[aid]
        if atype == 'cap':
            return 1
        elif atype == 'reg':
            return raw_env.reg_act_num // 2
        else:
            return (np.array([0.0], dtype=np.float32)
                    if raw_env.bat_act_num == np.inf
                    else raw_env.bat_act_num // 2)

    episode_rewards = []
    episode_violations = []

    for ep in range(n_episodes):
        raw_env.seed(seed + ep)
        obs_dict = ma_env.reset()
        done = False
        ep_reward = 0
        ep_viol = []

        while not done:
            actions = {}
            for aid in agents:
                atype = agent_types[aid]
                if atype in models:
                    a, _ = models[atype].predict(
                        get_obs(aid, obs_dict), deterministic=True)
                    actions[aid] = map_action(aid, int(a))
                else:
                    actions[aid] = default_action(aid)

            obs_dict, reward_dict, done, info = ma_env.step(actions)
            ep_reward += np.mean(list(reward_dict.values()))
            if isinstance(info, dict) and 'constraint_cost' in info:
                ep_viol.append(info['constraint_cost'])

        episode_rewards.append(ep_reward)
        if ep_viol:
            episode_violations.append(np.mean(ep_viol))

    return {
        "reward_mean": float(np.mean(episode_rewards)),
        "reward_std": float(np.std(episode_rewards)),
        "violation_mean": float(np.mean(episode_violations)) if episode_violations else 0,
    }


# ============================================================
# MAIN RUNNER
# ============================================================

def run_single_ablation(config_name, config, env_name, steps, seed, output_dir):
    """Train and evaluate a single ablation configuration."""
    print(f"\n  --- {config_name}: {config['description']} ---")

    model_name = f"{output_dir}/ablation_{config_name}_{env_name}_seed{seed}"

    if not config['use_sharing']:
        # Separate policies per type
        type_models, train_time = train_no_sharing(
            env_name, steps, seed, model_name, verbose=0)
        training_data = {
            "timesteps": [], "rewards": [], "episode_rewards": [],
            "total_episodes": 0, "training_time_seconds": train_time
        }
    else:
        # Shared policy with ablation flags
        env = AblationIPPOWrapper(
            env_name,
            use_type_embed=config['use_type_embed'],
            use_global_obs=config['use_global_obs']
        )
        log_dir = f"logs/ablation_{config_name}_{env_name}_seed{seed}/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir + "monitor.csv")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        model = PPO(
            "MlpPolicy", env, device="cpu",
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, seed=seed, verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        )

        callback = TrainingMetricsCallback(log_freq=500)
        start = datetime.now()
        model.learn(total_timesteps=steps, callback=callback)
        train_time = (datetime.now() - start).total_seconds()

        model.save(model_name)
        training_data = callback.get_training_data()
        training_data["training_time_seconds"] = train_time
        env.close()

    # --- Evaluate ---
    # NOTE: the environment is deterministic and pinned to load profile 000
    # (ma_env.reset -> raw_env.reset(load_profile_idx=0); seed does not affect
    # the profile). Episodes are therefore identical within a seed, so we
    # evaluate a single episode. Reported uncertainty comes from the spread
    # ACROSS the 5 training seeds, not across episodes.
    if not config['use_sharing']:
        # Roll out the 3 type-specific policies jointly.
        eval_result = evaluate_no_sharing(env_name, type_models, seed, n_episodes=1)
        return training_data, eval_result

    eval_env = AblationIPPOWrapper(
        env_name,
        use_type_embed=config['use_type_embed'],
        use_global_obs=config['use_global_obs']
    )

    if config['use_sharing'] and os.path.exists(model_name + ".zip"):
        model = PPO.load(model_name)
        episode_rewards = []
        episode_violations = []

        for ep in range(1):  # deterministic, fixed profile 000 — 1 episode suffices
            obs, _ = eval_env.reset(seed=seed + ep)
            done = False
            ep_reward = 0
            ep_viol = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = eval_env.step(action)
                done = term or trunc
                ep_reward += reward
                if isinstance(info, dict) and 'constraint_cost' in info:
                    ep_viol.append(info['constraint_cost'])

            episode_rewards.append(ep_reward)
            if ep_viol:
                episode_violations.append(np.mean(ep_viol))

        eval_result = {
            "reward_mean": float(np.mean(episode_rewards)),
            "reward_std": float(np.std(episode_rewards)),
            "violation_mean": float(np.mean(episode_violations)) if episode_violations else 0,
        }
    else:
        eval_result = {
            "reward_mean": float(np.mean(training_data.get("episode_rewards", [0])[-10:])),
            "reward_std": 0.0,
            "violation_mean": 0.0,
        }

    eval_env.close()
    return training_data, eval_result


# Display order and labels for the paper table. no_sharing_no_embed is omitted
# on purpose: it is identical to no_sharing by construction (within a
# type-specific policy the type_id is a constant input, so zeroing it is a no-op).
SUMMARY_ORDER = [
    ('full_model',    'Full Specialist (Ours)'),
    ('no_sharing',    r'\quad w/o parameter sharing'),
    ('no_type_embed', r'\quad w/o type embedding'),
]
SUMMARY_DROP_FOOTNOTE = (
    r'\textsuperscript{\dag} The no-sharing $+$ no-embedding variant produces '
    r'results identical to no-sharing by construction: within a type-specific '
    r'policy, the agent-type identifier $\tau_i$ is a constant input, so zeroing '
    r'it does not affect the learned policy or evaluation behaviour.'
)


def summarize_from_json(json_path):
    """Recompute the ablation table from a saved results JSON using IQM +
    bootstrap 95% CI (run_exp.py's methodology). No re-training required.

    Robust to the single-seed PPO divergence outliers seen on 34Bus: IQM
    discards the top/bottom 25% of seeds, so a collapsed seed does not flip
    the conclusion the way the raw mean does.
    """
    with open(json_path) as f:
        results = json.load(f)

    # compute_statistics expects a 'loss_mean' on every eval result; the
    # ablation eval does not record loss, so inject 0.0 (loss is not part of
    # this table anyway).
    for cfg in results.values():
        for ev in cfg.get('eval_results', []):
            ev.setdefault('loss_mean', 0.0)

    stats = compute_statistics(results)

    print(f"\n{'='*78}")
    print(f"ABLATION RESULTS (IQM + bootstrap 95% CI) — from {os.path.basename(json_path)}")
    print(f"{'='*78}")
    print(f"{'Configuration':<28}{'IQM Reward':>12}  {'95% CI':<20}{'IQM Viol.':>10}  {'n'}")
    print('-'*78)
    for key, label in SUMMARY_ORDER:
        if key not in stats:
            continue
        s = stats[key]
        r, v = s['reward'], s['violation']
        ci = f"[{r['ci_95_lower']:.1f}, {r['ci_95_upper']:.1f}]"
        # interquartile mean of violations for consistency with reward
        viol_iqm = float(np.mean(np.sort(v['all'])[len(v['all'])//4: 3*len(v['all'])//4])) \
            if len(v['all']) >= 4 else v['mean']
        print(f"{label.replace(chr(92)+'quad','  ').replace(chr(92),''):<28}"
              f"{r['iqm']:>12.2f}  {ci:<20}{viol_iqm:>10.4f}  {s['n_seeds']}")

    print(f"\n--- LaTeX snippet ---")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"\textbf{Configuration} & \textbf{IQM Reward} $\uparrow$ & \textbf{95\% CI} \\")
    print(r"\hline")
    for key, label in SUMMARY_ORDER:
        if key not in stats:
            continue
        r = stats[key]['reward']
        print(f"{label} & ${r['iqm']:.2f}$ & $[{r['ci_95_lower']:.2f}, {r['ci_95_upper']:.2f}]$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(SUMMARY_DROP_FOOTNOTE)


def main():
    parser = argparse.ArgumentParser(description='Run Ablation Studies')
    parser.add_argument('--summarize', type=str, default=None,
                        help='Path to a saved ablation_*.json; recompute the '
                             'IQM/CI table and exit (no training).')
    parser.add_argument('--env_name', type=str, default='34Bus',
                        choices=['13Bus', '34Bus', '123Bus', '8500Node'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=None,
                        help='Override default timesteps')
    parser.add_argument('--output_dir', type=str, default='./ablation_results')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific configs to run (default: all)')
    args = parser.parse_args()

    if args.summarize:
        summarize_from_json(args.summarize)
        return

    default_steps = {'13Bus': 200000, '34Bus': 400000, '123Bus': 800000, '8500Node': 2000000}
    steps = args.steps or default_steps[args.env_name]
    seeds = SEEDS[:args.seeds]
    configs = args.configs or list(ABLATION_CONFIGS.keys())
    # Resolve output_dir to an absolute path BEFORE any env is built. PowerGym's
    # OpenDSS `Compile` chdir's into systems/<env>/, so a relative output_dir
    # would otherwise be created once here (empty) and again under the system
    # folder (with the real content). Anchoring it absolute keeps everything in
    # one predictable location regardless of cwd.
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# ABLATION STUDY")
    print(f"# Environment: {args.env_name}")
    print(f"# Configs: {configs}")
    print(f"# Seeds: {seeds}")
    print(f"# Steps: {steps}")
    print(f"{'#'*60}")

    all_results = {}

    for config_name in configs:
        config = ABLATION_CONFIGS[config_name]
        all_results[config_name] = {"eval_results": [], "training_data": []}

        for seed in seeds:
            print(f"\n[{config_name}] Seed {seed}")
            try:
                td, er = run_single_ablation(
                    config_name, config, args.env_name, steps, seed,
                    args.output_dir)
                all_results[config_name]["training_data"].append(td)
                all_results[config_name]["eval_results"].append(er)
                print(f"  Reward: {er['reward_mean']:.2f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # --- Print Results Table ---
    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS - IEEE {args.env_name}")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Reward (mean±std)':<25} {'Violation':<15} {'Train Time':<12}")
    print("-" * 77)

    for config_name in configs:
        evals = all_results[config_name]["eval_results"]
        tdata = all_results[config_name]["training_data"]
        if not evals:
            continue
        rewards = [e["reward_mean"] for e in evals]
        violations = [e["violation_mean"] for e in evals]
        times = [t.get("training_time_seconds", 0) for t in tdata]

        print(f"{config_name:<25} "
              f"{np.mean(rewards):>8.2f} ± {np.std(rewards):<8.2f}   "
              f"{np.mean(violations):>10.4f}   "
              f"{np.mean(times):>8.1f}s")

    # Save
    save_path = f"{args.output_dir}/ablation_{args.env_name}.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nResults saved: {save_path}")

    # LaTeX snippet
    print(f"\n--- LaTeX Table Snippet ---")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"\textbf{Configuration} & \textbf{Reward} $\uparrow$ & \textbf{Violation} $\downarrow$ \\")
    print(r"\hline")
    for config_name in configs:
        evals = all_results[config_name]["eval_results"]
        if not evals:
            continue
        rewards = [e["reward_mean"] for e in evals]
        violations = [e["violation_mean"] for e in evals]
        desc = ABLATION_CONFIGS[config_name]['description'].split('(')[0].strip()
        print(f"{desc} & ${np.mean(rewards):.2f} \\pm {np.std(rewards):.2f}$ & "
              f"${np.mean(violations):.4f}$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()