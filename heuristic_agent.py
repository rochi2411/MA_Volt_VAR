"""
heuristic_agent.py - Corrected Rule-Based Heuristic for PowerGym
==========================================================================
Implements Algorithm 1 from the paper with proper PowerGym API compatibility.

This replaces the broken random-action baseline with actual rule-based control.
"""

import numpy as np
import sys
sys.path.append('./powergym')

from powergym.env_register import make_env
import gymnasium as gym


# ============================================================
# GYMNASIUM COMPATIBILITY WRAPPER
# ============================================================

class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """Wraps old-style Gym environments to be compatible with Gymnasium API."""
    
    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            np.random.seed(seed)
        
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}
        
        self._last_obs = obs
        return obs, info
    
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        
        self._last_obs = obs
        return obs, reward, terminated, truncated, info


# ============================================================
# HEURISTIC CONTROL IMPLEMENTATION
# ============================================================

class HeuristicController:
    """
    Implements Algorithm 1: Rule-Based Volt-Var Control
    
    This controller extracts voltage information from observations and
    applies threshold-based control logic for each device type.
    """
    
    # Control parameters (from paper Algorithm 1)
    V_MIN = 0.95       # Lower voltage limit (p.u.)
    V_MAX = 1.05       # Upper voltage limit (p.u.)
    V_SET = 1.0        # Target voltage for regulators (p.u.)
    DEADBAND = 0.00625 # Regulator deadband (~0.75V on 120V base)
    
    def __init__(self, env_name: str):
        """
        Initialize controller with environment-specific configuration.
        
        Args:
            env_name: '13Bus', '34Bus', or '123Bus'
        """
        self.env_name = env_name
        
        # Get device configuration from PowerGym
        temp_env = make_env(env_name)
        
        # Extract action space structure
        self.action_space = temp_env.action_space
        
        # PowerGym stores device info in the environment
        # Action space is typically: [caps..., regs..., bats...]
        self._parse_env_structure(temp_env)
        
        temp_env.close()
        
        # Track current device states for stateful control
        self.current_tap_positions = {}
        self.current_cap_states = {}
    
    def _parse_env_structure(self, env):
        """
        Parse PowerGym environment to understand device structure.
        
        PowerGym action spaces are structured as:
        - Capacitors: Binary (0=OFF, 1=ON)
        - Regulators: Discrete (0 to N_taps-1)
        - Batteries: Continuous or Discrete
        """
        # Get counts from environment
        self.n_caps = getattr(env, 'cap_num', 0)
        self.n_regs = getattr(env, 'reg_num', 0) 
        self.n_bats = getattr(env, 'bat_num', 0)
        
        # If attributes not directly available, infer from action space
        if self.n_caps == 0 and self.n_regs == 0:
            self._infer_device_counts(env)
        
        # Regulator tap settings
        self.reg_act_num = getattr(env, 'reg_act_num', 33)  # Default: 0-32
        
        # Battery action configuration
        self.bat_act_num = getattr(env, 'bat_act_num', np.inf)  # Continuous if inf
        
        print(f"[HeuristicController] {self.env_name}: "
              f"{self.n_caps} caps, {self.n_regs} regs, {self.n_bats} bats")
    
    def _infer_device_counts(self, env):
        """Infer device counts from environment configuration."""
        # Environment-specific configurations based on IEEE test feeders
        configs = {
            '13Bus': {'caps': 2, 'regs': 3, 'bats': 1},   # 2 cap banks, 3 reg phases, 1 BESS
            '34Bus': {'caps': 2, 'regs': 6, 'bats': 2},   # 2 caps, 2x3-phase regs, 2 BESS
            '123Bus': {'caps': 4, 'regs': 7, 'bats': 4},  # 4 caps, 7 reg controls, 4 BESS
        }
        
        config = configs.get(self.env_name, {'caps': 2, 'regs': 3, 'bats': 1})
        self.n_caps = config['caps']
        self.n_regs = config['regs']
        self.n_bats = config['bats']
    
    def extract_voltages_from_obs(self, obs: np.ndarray) -> dict:
        """
        Extract voltage information from observation vector.
        
        PowerGym observations typically include:
        - Bus voltages (all phases)
        - Device states (tap positions, cap states, battery SoC)
        
        Returns:
            Dictionary with voltage statistics
        """
        # PowerGym obs structure varies, but voltages are typically first
        # Voltage values are in p.u. (0.8 to 1.2 range typically)
        
        # Heuristic: values in [0.8, 1.2] are likely voltages
        voltage_mask = (obs >= 0.7) & (obs <= 1.3)
        potential_voltages = obs[voltage_mask]
        
        if len(potential_voltages) > 0:
            return {
                'min': np.min(potential_voltages),
                'max': np.max(potential_voltages),
                'mean': np.mean(potential_voltages),
                'all': potential_voltages
            }
        else:
            # Fallback: assume first portion of obs are voltages
            n_buses = len(obs) // 3  # Rough estimate
            voltages = obs[:n_buses]
            return {
                'min': np.min(voltages) if len(voltages) > 0 else 1.0,
                'max': np.max(voltages) if len(voltages) > 0 else 1.0,
                'mean': np.mean(voltages) if len(voltages) > 0 else 1.0,
                'all': voltages
            }
    
    def get_action(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        """
        Compute heuristic control action based on Algorithm 1.
        
        Args:
            obs: Observation from environment
            info: Info dict from environment (may contain additional data)
        
        Returns:
            Action array compatible with PowerGym action space
        """
        # Extract voltage information
        voltages = self.extract_voltages_from_obs(obs)
        v_min = voltages['min']
        v_max = voltages['max']
        v_mean = voltages['mean']
        
        # Also check info dict for explicit voltage data
        if info and 'bus_voltages' in info:
            bus_voltages = info['bus_voltages']
            v_min = np.min(bus_voltages)
            v_max = np.max(bus_voltages)
            v_mean = np.mean(bus_voltages)
        
        # Initialize action list
        actions = []
        
        # ===== 1. CAPACITOR CONTROL (Binary: 0=OFF, 1=ON) =====
        for cap_idx in range(self.n_caps):
            if v_min < self.V_MIN:
                # Undervoltage: Switch capacitor ON to inject reactive power
                cap_action = 1
            elif v_max > self.V_MAX:
                # Overvoltage: Switch capacitor OFF
                cap_action = 0
            else:
                # Normal: Maintain previous state (default to ON for support)
                cap_action = self.current_cap_states.get(cap_idx, 1)
            
            actions.append(cap_action)
            self.current_cap_states[cap_idx] = cap_action
        
        # ===== 2. REGULATOR CONTROL (Discrete: 0 to reg_act_num-1) =====
        for reg_idx in range(self.n_regs):
            current_tap = self.current_tap_positions.get(reg_idx, 16)  # Default: mid-tap
            
            if v_mean < self.V_SET - self.DEADBAND:
                # Voltage too low: Tap UP to boost
                new_tap = min(current_tap + 1, self.reg_act_num - 1)
            elif v_mean > self.V_SET + self.DEADBAND:
                # Voltage too high: Tap DOWN to reduce
                new_tap = max(current_tap - 1, 0)
            else:
                # Within deadband: Maintain tap
                new_tap = current_tap
            
            actions.append(new_tap)
            self.current_tap_positions[reg_idx] = new_tap
        
        # ===== 3. BATTERY CONTROL =====
        for bat_idx in range(self.n_bats):
            if self.bat_act_num == np.inf:
                # Continuous action: -1.0 (charge) to 1.0 (discharge)
                if v_min < self.V_MIN:
                    bat_action = 0.5  # Discharge to support voltage
                elif v_max > self.V_MAX:
                    bat_action = -0.5  # Charge to absorb power
                else:
                    bat_action = 0.0  # Idle
                actions.append(bat_action)
            else:
                # Discrete action
                if v_min < self.V_MIN:
                    bat_action = self.bat_act_num - 1  # Max discharge
                elif v_max > self.V_MAX:
                    bat_action = 0  # Max charge
                else:
                    bat_action = self.bat_act_num // 2  # Idle (middle action)
                actions.append(int(bat_action))
        
        # Convert to numpy array with appropriate dtype
        action_array = np.array(actions, dtype=np.float32)
        
        # Handle action space compatibility
        if hasattr(self.action_space, 'shape'):
            # Ensure correct shape
            expected_size = self.action_space.shape[0] if len(self.action_space.shape) > 0 else 1
            if len(action_array) < expected_size:
                # Pad with zeros if needed
                action_array = np.pad(action_array, (0, expected_size - len(action_array)))
            elif len(action_array) > expected_size:
                # Truncate if needed
                action_array = action_array[:expected_size]
        
        return action_array
    
    def reset(self):
        """Reset controller state for new episode."""
        self.current_tap_positions = {i: 16 for i in range(self.n_regs)}  # Mid-tap
        self.current_cap_states = {i: 0 for i in range(self.n_caps)}  # Start OFF


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_heuristic(env_name: str, n_episodes: int = 10, seed: int = None) -> dict:
    """
    Evaluate rule-based heuristic baseline using proper Algorithm 1 logic.
    
    Args:
        env_name: Environment name ('13Bus', '34Bus', '123Bus')
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    raw_env = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw_env)
    
    # Create heuristic controller
    controller = HeuristicController(env_name)
    
    # Metrics storage
    episode_rewards = []
    episode_violations = []
    episode_losses = []
    
    for ep in range(n_episodes):
        # Reset
        if seed is not None:
            obs, info = env.reset(seed=seed + ep)
        else:
            obs, info = env.reset()
        
        controller.reset()
        
        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []
        step_count = 0
        
        while not done:
            # Get heuristic action (ACTUAL RULE-BASED CONTROL)
            action = controller.get_action(obs, info)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            step_count += 1
            
            # Extract metrics from info
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
        
        print(f"  Episode {ep+1}/{n_episodes}: Reward={ep_reward:.2f}, "
              f"Steps={step_count}, Violations={np.mean(ep_violations) if ep_violations else 0:.4f}")
    
    env.close()
    
    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_all": episode_rewards,
        "violation_mean": np.mean(episode_violations) if episode_violations else 0,
        "violation_std": np.std(episode_violations) if episode_violations else 0,
        "loss_mean": np.mean(episode_losses) if episode_losses else 0,
        "loss_std": np.std(episode_losses) if episode_losses else 0,
        "n_episodes": n_episodes
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Heuristic Baseline')
    parser.add_argument('--env', type=str, default='13Bus', 
                        choices=['13Bus', '34Bus', '123Bus'])
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Evaluating Heuristic Baseline on {args.env}")
    print(f"{'='*60}\n")
    
    results = evaluate_heuristic(args.env, args.episodes, args.seed)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Reward: {results['reward_mean']:.2f} ± {results['reward_std']:.2f}")
    print(f"  Voltage Violation: {results['violation_mean']:.4f} ± {results['violation_std']:.4f}")
    print(f"  Power Loss: {results['loss_mean']:.4f} ± {results['loss_std']:.4f}")
    print(f"{'='*60}\n")