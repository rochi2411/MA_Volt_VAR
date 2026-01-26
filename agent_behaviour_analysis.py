"""
analyze_agent_behavior_fixed.py - Fixed Agent Behavior Analysis
================================================================
Generates publication-quality plots using REAL data from trained models.

Key Fixes:
1. Extracts actual actions taken by the agent (not synthetic data)
2. Records real observations from PowerGym
3. Compares Specialist vs Monolithic vs Heuristic behavior
4. Proper robustness analysis with statistical measures

Usage:
    python analyze_agent_behavior_fixed.py --env_name 13Bus
    python analyze_agent_behavior_fixed.py --env_name all --robustness
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from scipy import stats


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

sys.path.append('./powergym')

from stable_baselines3 import PPO
from powergym.env_register import make_env
from train_marl import IPPO_Wrapper
import gymnasium as gym

# Import heuristic controller (try both possible module names)
try:
    from heuristic_agent import HeuristicController
except ImportError:
    print("WARNING: Could not import HeuristicController. Heuristic analysis will be skipped.")
    HeuristicController = None

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """Gymnasium API compatibility wrapper."""
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result


# ============================================================
# DEVICE CONFIGURATION PER SYSTEM
# ============================================================

DEVICE_CONFIG = {
    '13Bus': {
        'n_caps': 2,
        'n_regs': 3,  # 3 single-phase regulators
        'n_bats': 1,
        'cap_names': ['Cap 675 (3-Ph)', 'Cap 611 (1-Ph)'],
        'reg_names': ['Reg 1 (Ph A)', 'Reg 2 (Ph B)', 'Reg 3 (Ph C)'],
        'bat_names': ['BESS 680'],
    },
    '34Bus': {
        'n_caps': 2,
        'n_regs': 6,  # 2 regulators × 3 phases
        'n_bats': 2,
        'cap_names': ['Cap 844', 'Cap 848'],
        'reg_names': ['Reg1 Ph-A', 'Reg1 Ph-B', 'Reg1 Ph-C', 'Reg2 Ph-A', 'Reg2 Ph-B', 'Reg2 Ph-C'],
        'bat_names': ['BESS 890', 'BESS 832'],
    },
    '123Bus': {
        'n_caps': 4,
        'n_regs': 7,  # Complex configuration
        'n_bats': 4,
        'cap_names': ['C83 (3-Ph)', 'C88 (Ph-A)', 'C90 (Ph-B)', 'C92 (Ph-C)'],
        'reg_names': ['Reg1 (Gang)', 'Reg2 (Ph-A)', 'Reg3 (Ph-A)', 'Reg3 (Ph-C)', 
                      'Reg4 (Ph-A)', 'Reg4 (Ph-B)', 'Reg4 (Ph-C)'],
        'bat_names': ['BESS 33', 'BESS 114', 'BESS 67', 'BESS 300'],
    }
}


# ============================================================
# DATA COLLECTION FROM REAL MODELS
# ============================================================

def collect_agent_actions(env_name: str, model, agent_type: str, 
                          n_steps: int = 24, seed: int = 42) -> dict:
    """
    Collect ACTUAL actions taken by the agent during an episode.
    
    Args:
        env_name: Environment name
        model: Trained model (PPO) or HeuristicController
        agent_type: 'specialist', 'monolithic', or 'heuristic'
        n_steps: Number of environment steps (24 = 4 hours at 10-min intervals)
        seed: Random seed
    
    Returns:
        Dictionary with timestep-by-timestep data
    """
    config = DEVICE_CONFIG[env_name]
    
    # Create appropriate environment
    if agent_type == 'specialist':
        env = IPPO_Wrapper(env_name)
    else:
        raw_env = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw_env)
    
    # Initialize data storage
    data = {
        'timesteps': [],
        'hours': [],
        'rewards': [],
        'violations': [],
        'power_loss': [],
        'observations': [],
        # Device-specific actions
        'cap_actions': {f'cap_{i}': [] for i in range(config['n_caps'])},
        'reg_actions': {f'reg_{i}': [] for i in range(config['n_regs'])},
        'bat_actions': {f'bat_{i}': [] for i in range(config['n_bats'])},
        # Voltage data
        'voltage_min': [],
        'voltage_max': [],
        'voltage_mean': [],
    }
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # For heuristic, create controller
    if agent_type == 'heuristic':
        if HeuristicController is None:
            print("    ERROR: HeuristicController not available")
            return data
        controller = HeuristicController(env_name)
        controller.reset()
    
    # For SPECIALIST: Need to collect actions from sequential agent processing
    if agent_type == 'specialist':
        # The IPPO wrapper processes agents sequentially
        # Each step() call handles one agent, returns intermediate obs until all agents acted
        
        # Get agent list and types from environment
        agent_list = env.agents  # List of agent IDs
        n_agents = len(agent_list)
        
        # Get agent types from underlying multi-agent env
        agent_types_dict = env.ma_env.agent_types  # {agent_id: 'cap'/'reg'/'bat'}
        
        # Count devices of each type to create index mapping
        cap_count, reg_count, bat_count = 0, 0, 0
        agent_to_idx = {}  # Map agent_id to device index
        for agent_id in agent_list:
            atype = agent_types_dict[agent_id]
            if atype == 'cap':
                agent_to_idx[agent_id] = ('cap', cap_count)
                cap_count += 1
            elif atype == 'reg':
                agent_to_idx[agent_id] = ('reg', reg_count)
                reg_count += 1
            elif atype == 'bat':
                agent_to_idx[agent_id] = ('bat', bat_count)
                bat_count += 1
        
        print(f"    IPPO environment has {n_agents} agents: {cap_count} caps, {reg_count} regs, {bat_count} bats")
        
        episode_done = False  # Track episode termination separately
        
        for step in range(n_steps):
            if episode_done:
                break
                
            data['timesteps'].append(step)
            data['hours'].append(step / 6.0)
            
            # Collect actions for all agents in this timestep
            agent_actions = {}  # Store as {agent_id: action}
            final_reward = 0
            final_info = {}
            
            # Process each agent sequentially
            for agent_idx in range(n_agents):
                agent_id = agent_list[agent_idx]
                
                # Get action for current agent
                action, _ = model.predict(obs, deterministic=True)
                agent_actions[agent_id] = int(action)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Only the last agent's step returns the real reward/info/termination
                if agent_idx == n_agents - 1:
                    final_reward = reward
                    final_info = info
                    episode_done = terminated or truncated
            
            # Parse collected actions by device type using the mapping
            for agent_id, action in agent_actions.items():
                device_type, device_idx = agent_to_idx[agent_id]
                
                if device_type == 'cap':
                    key = f'cap_{device_idx}'
                    if key in data['cap_actions']:
                        data['cap_actions'][key].append(action % 2)
                elif device_type == 'reg':
                    key = f'reg_{device_idx}'
                    if key in data['reg_actions']:
                        data['reg_actions'][key].append(action % 33)
                elif device_type == 'bat':
                    key = f'bat_{device_idx}'
                    if key in data['bat_actions']:
                        # Convert discrete (0-32) to continuous (-1 to 1)
                        bat_continuous = (action / 16.0) - 1.0
                        data['bat_actions'][key].append(bat_continuous)
            
            # Fill in missing devices with defaults (in case config doesn't match)
            for i in range(config['n_caps']):
                if len(data['cap_actions'][f'cap_{i}']) < step + 1:
                    data['cap_actions'][f'cap_{i}'].append(0)
            for i in range(config['n_regs']):
                if len(data['reg_actions'][f'reg_{i}']) < step + 1:
                    data['reg_actions'][f'reg_{i}'].append(16)
            for i in range(config['n_bats']):
                if len(data['bat_actions'][f'bat_{i}']) < step + 1:
                    data['bat_actions'][f'bat_{i}'].append(0.0)
            
            # Record metrics from final step
            data['rewards'].append(final_reward)
            
            # Extract voltage info from last observation
            voltage_mask = (obs >= 0.7) & (obs <= 1.3)
            voltages = obs[voltage_mask] if np.any(voltage_mask) else np.array([1.0])
            data['voltage_min'].append(np.min(voltages))
            data['voltage_max'].append(np.max(voltages))
            data['voltage_mean'].append(np.mean(voltages))
            
            # Extract metrics from info
            if isinstance(final_info, dict):
                if 'constraint_cost' in final_info:
                    data['violations'].append(final_info['constraint_cost'])
                elif 'vol_reward' in final_info:
                    data['violations'].append(abs(final_info['vol_reward']))
                else:
                    data['violations'].append(0.0)
                
                if 'power_loss_ratio' in final_info:
                    data['power_loss'].append(final_info['power_loss_ratio'])
                else:
                    data['power_loss'].append(0.0)
            else:
                data['violations'].append(0.0)
                data['power_loss'].append(0.0)
    
    else:
        # MONOLITHIC and HEURISTIC: Single action covers all devices
        for step in range(n_steps):
            data['timesteps'].append(step)
            data['hours'].append(step / 6.0)
            data['observations'].append(obs.copy())
            
            # Get action
            if agent_type == 'heuristic':
                action = controller.get_action(obs, info)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Parse action into device-specific components
            action_flat = np.atleast_1d(action).flatten()
            
            idx = 0
            # Capacitor actions (binary)
            for i in range(config['n_caps']):
                if idx < len(action_flat):
                    cap_act = int(action_flat[idx]) % 2
                    data['cap_actions'][f'cap_{i}'].append(cap_act)
                    idx += 1
                else:
                    data['cap_actions'][f'cap_{i}'].append(0)
            
            # Regulator actions (discrete taps)
            for i in range(config['n_regs']):
                if idx < len(action_flat):
                    reg_act = int(action_flat[idx]) % 33
                    data['reg_actions'][f'reg_{i}'].append(reg_act)
                    idx += 1
                else:
                    data['reg_actions'][f'reg_{i}'].append(16)
            
            # Battery actions
            for i in range(config['n_bats']):
                if idx < len(action_flat):
                    bat_act = float(action_flat[idx])
                    if abs(bat_act) > 1:
                        bat_act = (bat_act / 16.0) - 1.0
                    data['bat_actions'][f'bat_{i}'].append(bat_act)
                    idx += 1
                else:
                    data['bat_actions'][f'bat_{i}'].append(0.0)
            
            # Extract voltage info from observation
            voltage_mask = (obs >= 0.7) & (obs <= 1.3)
            voltages = obs[voltage_mask] if np.any(voltage_mask) else np.array([1.0])
            data['voltage_min'].append(np.min(voltages))
            data['voltage_max'].append(np.max(voltages))
            data['voltage_mean'].append(np.mean(voltages))
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            data['rewards'].append(reward)
            
            # Extract metrics from info
            if isinstance(info, dict):
                if 'constraint_cost' in info:
                    data['violations'].append(info['constraint_cost'])
                elif 'vol_reward' in info:
                    data['violations'].append(abs(info['vol_reward']))
                else:
                    data['violations'].append(0.0)
                
                if 'power_loss_ratio' in info:
                    data['power_loss'].append(info['power_loss_ratio'])
                else:
                    data['power_loss'].append(0.0)
            else:
                data['violations'].append(0.0)
                data['power_loss'].append(0.0)
            
            if terminated or truncated:
                print(f"    Episode terminated at step {step} ({step/6:.1f} hours)")
                break
    
    env.close()
    
    # Convert to numpy arrays
    for key in ['timesteps', 'hours', 'rewards', 'violations', 'power_loss',
                'voltage_min', 'voltage_max', 'voltage_mean']:
        data[key] = np.array(data[key])
    
    return data


def collect_comparative_data(env_name: str, specialist_path: str, 
                             monolithic_path: str, seed: int = 42) -> dict:
    """
    Collect data from all three methods for comparison.
    """
    print(f"\nCollecting data for {env_name}...")
    
    results = {}
    
    def model_exists(path):
        """Check if model exists (with or without .zip)"""
        return os.path.exists(path + '.zip') or os.path.exists(path)
    
    # Specialist
    print("  Loading Specialist model...")
    if model_exists(specialist_path):
        try:
            model = PPO.load(specialist_path)
            results['specialist'] = collect_agent_actions(env_name, model, 'specialist', seed=seed)
            print(f"    Collected {len(results['specialist']['timesteps'])} steps")
        except Exception as e:
            print(f"    ERROR loading specialist: {e}")
    else:
        print(f"    WARNING: Specialist model not found at {specialist_path}")
    
    # Monolithic
    print("  Loading Monolithic model...")
    if model_exists(monolithic_path):
        try:
            model = PPO.load(monolithic_path)
            results['monolithic'] = collect_agent_actions(env_name, model, 'monolithic', seed=seed)
            print(f"    Collected {len(results['monolithic']['timesteps'])} steps")
        except Exception as e:
            print(f"    ERROR loading monolithic: {e}")
    else:
        print(f"    WARNING: Monolithic model not found at {monolithic_path}")
    
    # Heuristic (always runs)
    print("  Running Heuristic controller...")
    try:
        results['heuristic'] = collect_agent_actions(env_name, None, 'heuristic', seed=seed)
        print(f"    Collected {len(results['heuristic']['timesteps'])} steps")
    except Exception as e:
        print(f"    ERROR running heuristic: {e}")
    
    return results


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_regulator_comparison(data_dict: dict, env_name: str, save_path: str):
    """
    Compare regulator tap positions across all methods.
    """
    config = DEVICE_CONFIG[env_name]
    n_regs = min(3, config['n_regs'])  # Show up to 3 regulators
    
    fig, axes = plt.subplots(n_regs, 1, figsize=(12, 3*n_regs), sharex=True)
    if n_regs == 1:
        axes = [axes]
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db', 'heuristic': '#e74c3c'}
    
    for i, ax in enumerate(axes):
        reg_key = f'reg_{i}'
        reg_name = config['reg_names'][i] if i < len(config['reg_names']) else f'Reg {i+1}'
        
        for method, data in data_dict.items():
            if reg_key in data['reg_actions']:
                hours = data['hours']
                taps = data['reg_actions'][reg_key]
                ax.plot(hours, taps, color=colors[method], 
                       label=method.capitalize(), linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(f'{reg_name}\nTap Position')
        ax.set_ylim(0, 32)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (Hours)')
    # Use actual episode duration
    for ax in axes:
        ax.set_xlim(0, 4)
    fig.suptitle(f'IEEE {env_name}: Regulator Control Comparison', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_capacitor_comparison(data_dict: dict, env_name: str, save_path: str):
    """
    Compare capacitor switching patterns across methods.
    """
    config = DEVICE_CONFIG[env_name]
    n_caps = config['n_caps']
    
    fig, axes = plt.subplots(n_caps, 1, figsize=(12, 2.5*n_caps), sharex=True)
    if n_caps == 1:
        axes = [axes]
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db', 'heuristic': '#e74c3c'}
    offsets = {'specialist': 0.0, 'monolithic': 0.15, 'heuristic': 0.30}
    
    for i, ax in enumerate(axes):
        cap_key = f'cap_{i}'
        cap_name = config['cap_names'][i] if i < len(config['cap_names']) else f'Cap {i+1}'
        
        for method, data in data_dict.items():
            if cap_key in data['cap_actions']:
                hours = data['hours']
                states = np.array(data['cap_actions'][cap_key]) + offsets[method]
                ax.step(hours, states, color=colors[method], 
                       label=method.capitalize(), linewidth=2, where='post', alpha=0.8)
        
        ax.set_ylabel(f'{cap_name}\nStatus')
        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OFF', 'ON'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (Hours)')
    # Use actual episode duration
    for ax in axes:
        ax.set_xlim(0, 4)
    fig.suptitle(f'IEEE {env_name}: Capacitor Switching Comparison', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_battery_comparison(data_dict: dict, env_name: str, save_path: str):
    """
    Compare battery dispatch strategies across methods.
    """
    config = DEVICE_CONFIG[env_name]
    n_bats = config['n_bats']
    
    fig, axes = plt.subplots(n_bats, 1, figsize=(12, 3*n_bats), sharex=True)
    if n_bats == 1:
        axes = [axes]
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db', 'heuristic': '#e74c3c'}
    
    for i, ax in enumerate(axes):
        bat_key = f'bat_{i}'
        bat_name = config['bat_names'][i] if i < len(config['bat_names']) else f'BESS {i+1}'
        
        for method, data in data_dict.items():
            if bat_key in data['bat_actions']:
                hours = data['hours']
                power = np.array(data['bat_actions'][bat_key])
                ax.plot(hours, power, color=colors[method], 
                       label=method.capitalize(), linewidth=1.5, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between([0, 24], 0, 1, alpha=0.1, color='green', label='Discharge Zone')
        ax.fill_between([0, 24], -1, 0, alpha=0.1, color='red', label='Charge Zone')
        
        ax.set_ylabel(f'{bat_name}\nPower (p.u.)')
        ax.set_ylim(-1.2, 1.2)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (Hours)')
    # Use actual episode duration
    for ax in axes:
        ax.set_xlim(0, 4)
    fig.suptitle(f'IEEE {env_name}: Battery Dispatch Comparison', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_voltage_profile_comparison(data_dict: dict, env_name: str, save_path: str):
    """
    Compare voltage profiles across methods over time.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db', 'heuristic': '#e74c3c'}
    labels = ['Min Voltage', 'Mean Voltage', 'Max Voltage']
    keys = ['voltage_min', 'voltage_mean', 'voltage_max']
    
    for ax, key, label in zip(axes, keys, labels):
        for method, data in data_dict.items():
            hours = data['hours']
            voltages = data[key]
            ax.plot(hours, voltages, color=colors[method], 
                   label=method.capitalize(), linewidth=1.5, alpha=0.8)
        
        ax.axhline(y=1.05, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.fill_between([0, 4], 0.95, 1.05, alpha=0.1, color='green')
        
        ax.set_ylabel(f'{label} (p.u.)')
        ax.set_ylim(0.90, 1.10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (Hours)')
    # Use actual episode duration
    for ax in axes:
        ax.set_xlim(0, 4)
    fig.suptitle(f'IEEE {env_name}: Voltage Profile Comparison', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cumulative_rewards(data_dict: dict, env_name: str, save_path: str):
    """
    Plot cumulative rewards over time for all methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db', 'heuristic': '#e74c3c'}
    
    for method, data in data_dict.items():
        hours = data['hours']
        cumulative_reward = np.cumsum(data['rewards'])
        ax.plot(hours, cumulative_reward, color=colors[method], 
               label=f"{method.capitalize()} (Final: {cumulative_reward[-1]:.1f})",
               linewidth=2)
    
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'IEEE {env_name}: Cumulative Reward Comparison')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    # Use actual episode duration
    ax.set_xlim(0, 4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_violation_heatmap(data_dict: dict, env_name: str, save_path: str):
    """
    Create heatmap of voltage violations over time.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    methods = ['specialist', 'monolithic', 'heuristic']
    titles = ['Specialist Ensemble', 'Monolithic PPO', 'Heuristic']
    
    for ax, method, title in zip(axes, methods, titles):
        if method in data_dict:
            violations = data_dict[method]['violations']
            hours = data_dict[method]['hours']
            
            # Use actual episode duration (4 hours)
            n_hours = 4
            bins = np.zeros((1, n_hours))
            for h, v in zip(hours, violations):
                hour_idx = min(int(h), n_hours - 1)
                bins[0, hour_idx] = max(bins[0, hour_idx], v)
            
            im = ax.imshow(bins, aspect='auto', cmap='RdYlGn_r', 
                          vmin=0, vmax=max(0.1, np.max(bins)),
                          extent=[0, 4, 0, 1])
            ax.set_xlabel('Hour')
            ax.set_title(f'{title}\nTotal: {np.sum(violations):.3f}')
            ax.set_yticks([])
            ax.set_xlim(0, 4)
            plt.colorbar(im, ax=ax, label='Violation Index')
    
    fig.suptitle(f'IEEE {env_name}: Voltage Violation Heatmap', fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

def run_robustness_analysis(env_name: str, specialist_path: str, 
                            uncertainties: list = [0, 5, 10, 15, 20, 25],
                            n_episodes: int = 5) -> dict:
    """
    Test agent robustness under observation noise.
    """
    print(f"\nRunning robustness analysis for {env_name}...")
    
    results = {
        'uncertainty': uncertainties,
        'specialist': {'reward_mean': [], 'reward_std': [], 'violation_mean': [], 'violation_std': []},
        'heuristic': {'reward_mean': [], 'reward_std': [], 'violation_mean': [], 'violation_std': []},
        'model_found': False  # Track if model was found
    }
    
    # Load specialist model
    model_path_zip = specialist_path + '.zip'
    if not os.path.exists(model_path_zip):
        # Try alternate locations
        alt_paths = [
            specialist_path,  # Without .zip
            f"./systems/{env_name}/experiment_results_fixed/specialist_{env_name}_seed42",
            f"./experiment_results/specialist_{env_name}_seed42",
        ]
        
        model_path_zip = None
        for alt in alt_paths:
            if os.path.exists(alt + '.zip'):
                model_path_zip = alt + '.zip'
                specialist_path = alt
                print(f"  Found model at alternate location: {alt}")
                break
            elif os.path.exists(alt):
                model_path_zip = alt
                specialist_path = alt
                print(f"  Found model at alternate location: {alt}")
                break
        
        if model_path_zip is None:
            print(f"  WARNING: Model not found. Tried paths:")
            for alt in alt_paths:
                print(f"    - {alt}")
            return results
    
    results['model_found'] = True
    model = PPO.load(specialist_path)
    
    for unc in uncertainties:
        print(f"  Testing with {unc}% noise...")
        
        specialist_rewards = []
        specialist_violations = []
        heuristic_rewards = []
        heuristic_violations = []
        
        for ep in range(n_episodes):
            # Test specialist
            env = IPPO_Wrapper(env_name)
            obs, _ = env.reset(seed=42 + ep)
            
            ep_reward = 0
            ep_violations = []
            done = False
            
            while not done:
                # Add noise to observation
                if unc > 0:
                    noise = (unc / 100) * np.random.randn(*obs.shape)
                    noisy_obs = obs + noise
                else:
                    noisy_obs = obs
                
                action, _ = model.predict(noisy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                if info and 'constraint_cost' in info:
                    ep_violations.append(info['constraint_cost'])
            
            specialist_rewards.append(ep_reward)
            specialist_violations.append(np.mean(ep_violations) if ep_violations else 0)
            env.close()
            
            # Test heuristic
            raw_env = make_env(env_name)
            env = GymnasiumCompatibilityWrapper(raw_env)
            controller = HeuristicController(env_name)
            
            obs, info = env.reset(seed=42 + ep)
            controller.reset()
            
            ep_reward = 0
            ep_violations = []
            done = False
            
            while not done:
                if unc > 0:
                    noise = (unc / 100) * np.random.randn(*obs.shape)
                    noisy_obs = obs + noise
                else:
                    noisy_obs = obs
                
                action = controller.get_action(noisy_obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                if info and 'constraint_cost' in info:
                    ep_violations.append(info['constraint_cost'])
            
            heuristic_rewards.append(ep_reward)
            heuristic_violations.append(np.mean(ep_violations) if ep_violations else 0)
            env.close()
        
        # Store results
        results['specialist']['reward_mean'].append(np.mean(specialist_rewards))
        results['specialist']['reward_std'].append(np.std(specialist_rewards))
        results['specialist']['violation_mean'].append(np.mean(specialist_violations))
        results['specialist']['violation_std'].append(np.std(specialist_violations))
        
        results['heuristic']['reward_mean'].append(np.mean(heuristic_rewards))
        results['heuristic']['reward_std'].append(np.std(heuristic_rewards))
        results['heuristic']['violation_mean'].append(np.mean(heuristic_violations))
        results['heuristic']['violation_std'].append(np.std(heuristic_violations))
    
    return results


def plot_robustness_analysis(results: dict, env_name: str, save_path: str):
    """
    Plot robustness analysis results.
    """
    # Check if we have data to plot
    if not results.get('model_found', False) or len(results['specialist']['reward_mean']) == 0:
        print(f"  Skipping robustness plot for {env_name} - no data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    uncertainties = results['uncertainty']
    
    # Reward plot
    ax1 = axes[0]
    
    specialist_reward = results['specialist']['reward_mean']
    specialist_std = results['specialist']['reward_std']
    ax1.plot(uncertainties, specialist_reward, 'g-o', linewidth=2, markersize=8, 
            label='Specialist Ensemble')
    ax1.fill_between(uncertainties, 
                     np.array(specialist_reward) - np.array(specialist_std),
                     np.array(specialist_reward) + np.array(specialist_std),
                     color='green', alpha=0.2)
    
    heuristic_reward = results['heuristic']['reward_mean']
    heuristic_std = results['heuristic']['reward_std']
    ax1.plot(uncertainties, heuristic_reward, 'r--s', linewidth=2, markersize=8,
            label='Heuristic')
    ax1.fill_between(uncertainties,
                     np.array(heuristic_reward) - np.array(heuristic_std),
                     np.array(heuristic_reward) + np.array(heuristic_std),
                     color='red', alpha=0.2)
    
    ax1.set_xlabel('Observation Noise (%)')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Reward Degradation Under Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violation plot
    ax2 = axes[1]
    
    specialist_viol = results['specialist']['violation_mean']
    specialist_viol_std = results['specialist']['violation_std']
    ax2.plot(uncertainties, specialist_viol, 'g-o', linewidth=2, markersize=8,
            label='Specialist Ensemble')
    ax2.fill_between(uncertainties,
                     np.array(specialist_viol) - np.array(specialist_viol_std),
                     np.array(specialist_viol) + np.array(specialist_viol_std),
                     color='green', alpha=0.2)
    
    heuristic_viol = results['heuristic']['violation_mean']
    heuristic_viol_std = results['heuristic']['violation_std']
    ax2.plot(uncertainties, heuristic_viol, 'r--s', linewidth=2, markersize=8,
            label='Heuristic')
    ax2.fill_between(uncertainties,
                     np.array(heuristic_viol) - np.array(heuristic_viol_std),
                     np.array(heuristic_viol) + np.array(heuristic_viol_std),
                     color='red', alpha=0.2)
    
    ax2.set_xlabel('Observation Noise (%)')
    ax2.set_ylabel('Voltage Violation Index')
    ax2.set_title('Violation Increase Under Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'IEEE {env_name}: Robustness Analysis', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# TRAINING CURVE PLOTS
# ============================================================

def plot_training_curves_from_files(env_name: str, results_dir: str, 
                                    seeds: list, save_path: str):
    """
    Generate training curve plots from saved training data files.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'specialist': '#2ecc71', 'monolithic': '#3498db'}
    
    for method in ['specialist', 'monolithic']:
        all_rewards = []
        max_len = 0
        
        for seed in seeds:
            data_path = f"{results_dir}/{method}_{env_name}_seed{seed}_training_data.json"
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                rewards = data.get('rewards', [])
                if len(rewards) > 0:
                    all_rewards.append(rewards)
                    max_len = max(max_len, len(rewards))
        
        if len(all_rewards) == 0:
            continue
        
        # Pad to same length
        padded = []
        for r in all_rewards:
            if len(r) < max_len:
                r = r + [r[-1]] * (max_len - len(r))
            padded.append(r[:max_len])
        
        rewards_array = np.array(padded)
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        
        timesteps = np.arange(len(mean_rewards)) * 500  # Assuming 500-step logging
        
        ax.plot(timesteps, mean_rewards, color=colors[method],
               label=f"{method.capitalize()} (n={len(all_rewards)})", linewidth=2)
        ax.fill_between(timesteps,
                       mean_rewards - std_rewards,
                       mean_rewards + std_rewards,
                       color=colors[method], alpha=0.2)
    
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title(f'IEEE {env_name}: Training Convergence')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Agent Behavior Analysis')
    parser.add_argument('--env_name', type=str, default='13Bus',
                       choices=['13Bus', '34Bus', '123Bus', 'all'])
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='./analysis_plots')
    parser.add_argument('--robustness', action='store_true',
                       help='Run robustness analysis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1011',
                       help='Comma-separated list of seeds for training curves')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir.lstrip('./'))
    output_dir = os.path.join(script_dir, args.output_dir.lstrip('./'))
    os.makedirs(output_dir, exist_ok=True)
    
    environments = ['13Bus', '34Bus', '123Bus'] if args.env_name == 'all' else [args.env_name]
    
    print(f"\n{'='*60}")
    print(f"Agent Behavior Analysis")
    print(f"Environments: {environments}")
    print(f"Results Dir: {results_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"{'='*60}")
    
    for env_name in environments:
        print(f"\n{'#'*40}")
        print(f"# Processing {env_name}")
        print(f"{'#'*40}")
        
        # Paths to models - try multiple locations
        def find_model_path(model_type: str, env: str, seed: int, base_dirs: list) -> str:
            """Search for model in multiple possible directories."""
            filename = f"{model_type}_{env}_seed{seed}"
            for base in base_dirs:
                for path in [
                    f"{base}/{filename}",
                    f"{base}/{env}/{filename}",
                    f"./systems/{env}/{base.split('/')[-1]}/{filename}",
                ]:
                    if os.path.exists(path + '.zip') or os.path.exists(path):
                        return path
            return f"{base_dirs[0]}/{filename}"  # Default fallback
        
        search_dirs = [results_dir, './experiment_results_fixed', './experiment_results']
        specialist_path = find_model_path('specialist', env_name, args.seed, search_dirs)
        monolithic_path = find_model_path('monolithic', env_name, args.seed, search_dirs)
        
        print(f"  Specialist path: {specialist_path}")
        print(f"  Monolithic path: {monolithic_path}")
        
        # Collect comparative data
        data = collect_comparative_data(env_name, specialist_path, monolithic_path, args.seed)
        
        if len(data) == 0:
            print(f"  No data collected for {env_name}, skipping plots")
            continue
        
        # Generate plots
        print(f"\nGenerating plots for {env_name}...")
        
        # 1. Regulator comparison
        plot_regulator_comparison(
            data, env_name,
            save_path=f"{output_dir}/regulator_comparison_{env_name}.png"
        )
        
        # 2. Capacitor comparison
        plot_capacitor_comparison(
            data, env_name,
            save_path=f"{output_dir}/capacitor_comparison_{env_name}.png"
        )
        
        # 3. Battery comparison
        plot_battery_comparison(
            data, env_name,
            save_path=f"{output_dir}/battery_comparison_{env_name}.png"
        )
        
        # 4. Voltage profile comparison
        plot_voltage_profile_comparison(
            data, env_name,
            save_path=f"{output_dir}/voltage_profile_{env_name}.png"
        )
        
        # 5. Cumulative rewards
        plot_cumulative_rewards(
            data, env_name,
            save_path=f"{output_dir}/cumulative_rewards_{env_name}.png"
        )
        
        # 6. Violation heatmap
        plot_violation_heatmap(
            data, env_name,
            save_path=f"{output_dir}/violation_heatmap_{env_name}.png"
        )
        
        # 7. Training curves (from files)
        plot_training_curves_from_files(
            env_name, results_dir, seeds,
            save_path=f"{output_dir}/training_curve_{env_name}.png"
        )
        
        # 8. Robustness analysis (optional)
        if args.robustness:
            robustness_results = run_robustness_analysis(
                env_name, specialist_path,
                uncertainties=[0, 5, 10, 15, 20, 25],
                n_episodes=5
            )
            
            # Only plot and save if we have data
            if robustness_results.get('model_found', False):
                plot_robustness_analysis(
                    robustness_results, env_name,
                    save_path=f"{output_dir}/robustness_{env_name}.png"
                )
                
                # Save robustness data
                with open(f"{output_dir}/robustness_data_{env_name}.json", 'w') as f:
                    json.dump(robustness_results, f, indent=2, cls=NumpyEncoder)
            else:
                print(f"  Skipping robustness save for {env_name} - model not found")
        
        # Save collected data
        with open(f"{output_dir}/behavior_data_{env_name}.json", 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()