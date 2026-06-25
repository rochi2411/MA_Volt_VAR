"""
opendss_auto_baseline.py - Model-Based Optimal Control Baselines (v2)
======================================================================
Two model-based baselines for comparison with RL methods:

1. OpenDSS Auto-Control: Uses OpenDSS's built-in RegControl/CapControl
   for regulators and capacitors, with ADAPTIVE battery droop that
   calibrates to each feeder's voltage characteristics.

2. Greedy Voltage Optimization: At each step, evaluates the current 
   voltage state and greedily selects actions that minimize voltage
   violations — no perturbation or SOCP solver needed.

Usage:
    python opendss_auto_baseline.py --env_name 13Bus --episodes 10
    
Integration with run_exp.py:
    from opendss_auto_baseline import evaluate_opendss_auto, evaluate_socp_opf
"""

import numpy as np
import sys
import os
import time
import argparse

sys.path.append('./powergym')
from powergym.env_register import make_env
from heuristic_agent import GymnasiumCompatibilityWrapper


# ============================================================
# BASELINE 1: OpenDSS Auto-Control with Adaptive Battery Droop
# ============================================================

class OpenDSSAutoController:
    """
    Model-Based Voltage Control using OpenDSS's built-in controllers
    with adaptive battery dispatch.

    Improvements over v1:
    - Calibrates droop thresholds to each feeder's voltage profile
    - SoC-aware dispatch prevents over-discharge/charge
    - Reduced dispatch aggressiveness to avoid harming long-feeder systems
    """

    def __init__(self, env):
        self.env = env
        self.cap_num = env.cap_num
        self.reg_num = env.reg_num
        self.bat_num = env.bat_num
        self.bat_act_num = env.bat_act_num

        # Per-battery voltage baselines (calibrated on first call)
        self.v_baselines = {}
        self.calibrated = False

        # Conservative dispatch: only act on significant violations
        self.deadband_margin = 0.02  # +/-0.02 p.u. around baseline
        self.max_dispatch = 0.5     # Never more than 50% of rated power

    def _get_bus_voltage(self, bus_name):
        """Get average per-phase voltage magnitude at a bus."""
        v_data = self.env.circuit.bus_voltage(bus_name)
        v_mags = [v_data[i] for i in range(len(v_data)) if i % 2 == 0]
        return np.mean(v_mags) if v_mags else 1.0

    def calibrate(self):
        """
        Learn the baseline voltage at each battery bus.
        Called once after the first power flow solve.
        This adapts the droop to the feeder's natural voltage profile.
        """
        for bat_name in self.env.bat_names:
            bat = self.env.circuit.batteries[bat_name]
            v = self._get_bus_voltage(bat.bus1)
            self.v_baselines[bat_name] = v
        self.calibrated = True

    def _adaptive_droop(self, voltage, bat_name, soc):
        """
        Adaptive droop that uses per-bus voltage baseline.

        Key differences from fixed IEEE 1547 droop:
        1. Thresholds are relative to the bus baseline, not fixed at 1.0
        2. SoC-aware: reduces dispatch as battery depletes/fills
        3. Conservative: caps dispatch at max_dispatch to avoid harm
        """
        baseline = self.v_baselines.get(bat_name, 1.0)

        # Adaptive thresholds relative to this bus's baseline
        v_low = baseline - self.deadband_margin
        v_high = baseline + self.deadband_margin
        v_critical_low = 0.95   # ANSI lower limit
        v_critical_high = 1.05  # ANSI upper limit

        if voltage < v_critical_low:
            dispatch = self.max_dispatch
        elif voltage < v_low:
            dispatch = self.max_dispatch * (v_low - voltage) / (v_low - v_critical_low + 1e-6)
            dispatch = min(dispatch, self.max_dispatch)
        elif voltage > v_critical_high:
            dispatch = -self.max_dispatch
        elif voltage > v_high:
            dispatch = -self.max_dispatch * (voltage - v_high) / (v_critical_high - v_high + 1e-6)
            dispatch = max(dispatch, -self.max_dispatch)
        else:
            dispatch = 0.0

        # SoC-aware limiting
        if dispatch > 0 and soc < 0.15:
            dispatch *= soc / 0.15
        elif dispatch < 0 and soc > 0.90:
            dispatch *= (1.0 - soc) / 0.10

        return np.clip(dispatch, -self.max_dispatch, self.max_dispatch)

    def get_battery_actions(self):
        """Compute adaptive battery dispatch actions."""
        if not self.calibrated:
            self.calibrate()

        bat_actions = []

        for bat_name in self.env.bat_names:
            bat = self.env.circuit.batteries[bat_name]
            v_local = self._get_bus_voltage(bat.bus1)
            soc = bat.soc

            dispatch = self._adaptive_droop(v_local, bat_name, soc)

            if self.bat_act_num == np.inf:
                bat_actions.append(float(dispatch))
            else:
                mid = self.bat_act_num // 2
                action = mid + int(round(dispatch * mid))
                action = max(0, min(self.bat_act_num - 1, action))
                bat_actions.append(action)

        return bat_actions


def evaluate_opendss_auto(env_name: str, n_episodes: int = 10,
                           seed: int = None) -> dict:
    """Evaluate OpenDSS auto-control with adaptive battery droop."""
    raw_env = make_env(env_name, dss_act=True)
    controller = OpenDSSAutoController(raw_env)

    episode_rewards = []
    episode_violations = []
    episode_losses = []

    total_steps = 0
    inference_start = time.time()

    for ep in range(n_episodes):
        if seed is not None:
            raw_env.reset(load_profile_idx=(seed + ep) % raw_env.num_profiles)
        else:
            raw_env.reset(load_profile_idx=ep % raw_env.num_profiles)

        controller.calibrated = False  # Re-calibrate each episode
        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []

        while not done:
            bat_actions = controller.get_battery_actions()
            if raw_env.bat_num > 0:
                for i, bat_name in enumerate(raw_env.bat_names):
                    bat = raw_env.circuit.batteries[bat_name]
                    bat.step_before_solve(bat_actions[i])

            obs, reward, done, info = raw_env.dss_step()

            if raw_env.bat_num > 0:
                for bat_name in raw_env.bat_names:
                    bat = raw_env.circuit.batteries[bat_name]
                    bat.step_after_solve()
                bat_statuses = {
                    name: [bat.soc, -1 * bat.actual_power() / bat.max_kw]
                    for name, bat in raw_env.circuit.batteries.items()
                }
                raw_env.obs['bat_statuses'] = bat_statuses

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

    inference_time = time.time() - inference_start

    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_all": episode_rewards,
        "violation_mean": np.mean(episode_violations) if episode_violations else 0,
        "violation_std": np.std(episode_violations) if episode_violations else 0,
        "loss_mean": np.mean(episode_losses) if episode_losses else 0,
        "loss_std": np.std(episode_losses) if episode_losses else 0,
        "inference_time_total": inference_time,
        "inference_time_per_step": inference_time / max(total_steps, 1)
    }


# ============================================================
# BASELINE 2: Greedy Voltage Optimization 
# ============================================================

class GreedyVoltageOptimizer:
    """
    Greedy model-based controller that reads ALL bus voltages
    and selects actions to minimize voltage violations.

    Unlike the broken SOCP approach, this:
    - Reads actual voltages (no perturbation needed)
    - Uses simple rules with full network observability
    - Acts on the worst violations first (priority-based)
    - Is deterministic and fast

    Paper label: "Greedy Heuristic (myopic)" — a per-step, full-observability
    rule-based controller. NOTE: this is NOT the model-based OPF baseline. The
    defensible model-based OPF is the multi-period LinDist3Flow MILP in
    opf_lindist3flow.py (evaluate_oracle_opf / evaluate_mpc_opf); this greedy
    controller is retained only as a weaker myopic reference row.
    """

    def __init__(self, env):
        self.env = env
        self.cap_num = env.cap_num
        self.reg_num = env.reg_num
        self.bat_num = env.bat_num
        self.bat_act_num = env.bat_act_num
        self.reg_act_num = env.reg_act_num

    def _get_all_voltages(self):
        """Get min and max voltage across the entire network."""
        v_min = float('inf')
        v_max = float('-inf')
        v_sum = 0
        n = 0
        for bus_name in self.env.all_bus_names:
            v_data = self.env.circuit.bus_voltage(bus_name)
            v_mags = [v_data[i] for i in range(len(v_data)) if i % 2 == 0]
            for v in v_mags:
                if v > 0.01:
                    v_min = min(v_min, v)
                    v_max = max(v_max, v)
                    v_sum += v
                    n += 1
        v_avg = v_sum / max(n, 1)
        return v_min, v_max, v_avg

    def _get_bus_voltage(self, bus_name):
        v_data = self.env.circuit.bus_voltage(bus_name)
        v_mags = [v_data[i] for i in range(len(v_data)) if i % 2 == 0]
        return np.mean(v_mags) if v_mags else 1.0

    def get_action(self):
        """
        Compute globally-informed greedy action.
        
        Strategy:
        - Caps: ON if undervoltage nearby, OFF if overvoltage
        - Regs: Tap up/down to track 1.0 p.u. downstream
        - Bats: Discharge if system undervoltage, charge if overvoltage
        
        All decisions use GLOBAL voltage state (model-based advantage).
        """
        v_min, v_max, v_avg = self._get_all_voltages()
        actions = []

        # --- Capacitors ---
        for cap_name in self.env.cap_names:
            cap = self.env.circuit.capacitors[cap_name]
            v_local = self._get_bus_voltage(cap.bus1)

            if v_min < 0.95 or v_local < 0.96:
                actions.append(1)
            elif v_max > 1.05 or v_local > 1.04:
                actions.append(0)
            else:
                actions.append(int(cap.status))

        # --- Regulators ---
        for reg_name in self.env.reg_names:
            reg = self.env.circuit.regulators[reg_name]
            v_downstream = self._get_bus_voltage(reg.bus2)

            mintap, maxtap, numtaps = reg.tap_feature
            step = (maxtap - mintap) / numtaps
            current_tapnum = int(round((reg.tap - mintap) / step))

            if v_downstream < 0.96:
                tap_change = min(3, int((0.98 - v_downstream) / step) + 1)
                new_tapnum = current_tapnum + tap_change
            elif v_downstream > 1.04:
                tap_change = min(3, int((v_downstream - 1.02) / step) + 1)
                new_tapnum = current_tapnum - tap_change
            elif v_downstream < 0.98:
                new_tapnum = current_tapnum + 1
            elif v_downstream > 1.02:
                new_tapnum = current_tapnum - 1
            else:
                new_tapnum = current_tapnum

            new_tapnum = max(0, min(self.reg_act_num - 1, new_tapnum))
            actions.append(new_tapnum)

        # --- Batteries ---
        for bat_name in self.env.bat_names:
            bat = self.env.circuit.batteries[bat_name]
            v_local = self._get_bus_voltage(bat.bus1)
            soc = bat.soc

            if v_min < 0.95 and v_local < 0.98 and soc > 0.15:
                severity = min(1.0, (0.97 - v_min) / 0.05)
                dispatch = 0.3 + 0.4 * severity
            elif v_max > 1.05 and v_local > 1.02 and soc < 0.90:
                severity = min(1.0, (v_max - 1.03) / 0.05)
                dispatch = -(0.3 + 0.4 * severity)
            elif v_local < 0.95 and soc > 0.20:
                dispatch = 0.5
            elif v_local > 1.05 and soc < 0.85:
                dispatch = -0.5
            else:
                dispatch = 0.0

            # SoC protection
            if dispatch > 0:
                dispatch *= min(1.0, soc / 0.20)
            elif dispatch < 0:
                dispatch *= min(1.0, (1.0 - soc) / 0.15)

            if self.bat_act_num == np.inf:
                actions.append(float(np.clip(dispatch, -1, 1)))
            else:
                mid = self.bat_act_num // 2
                action = mid + int(round(dispatch * mid))
                action = max(0, min(self.bat_act_num - 1, action))
                actions.append(action)

        return np.array(actions)


def evaluate_socp_opf(env_name: str, n_episodes: int = 10,
                       seed: int = None) -> dict:
    """
    Evaluate Greedy Voltage Optimization baseline.

    A per-step greedy controller using full network observability.

    Paper label: "Greedy Heuristic (myopic)" (NOT the model-based OPF; see
    opf_lindist3flow.py for the LinDist3Flow MILP OPF baseline).
    """
    raw_env = make_env(env_name, dss_act=False)
    optimizer = GreedyVoltageOptimizer(raw_env)

    episode_rewards = []
    episode_violations = []
    episode_losses = []

    total_steps = 0
    inference_start = time.time()

    for ep in range(n_episodes):
        if seed is not None:
            raw_env.reset(load_profile_idx=(seed + ep) % raw_env.num_profiles)
        else:
            raw_env.reset(load_profile_idx=ep % raw_env.num_profiles)

        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []

        while not done:
            action = optimizer.get_action()
            obs, reward, done, info = raw_env.step(action)

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

    inference_time = time.time() - inference_start

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
        "solver": "Greedy Voltage Optimization (full observability)"
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Model-Based Control Baselines')
    parser.add_argument('--env_name', type=str, default='13Bus',
                        choices=['13Bus', '34Bus', '123Bus', '8500Node'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, default='both',
                        choices=['opendss', 'opf', 'both'])

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Model-Based Control Baselines")
    print(f"Environment: {args.env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"{'='*60}")

    if args.method in ['opendss', 'both']:
        print(f"\n--- OpenDSS Auto-Control + Adaptive Battery Droop ---")
        results = evaluate_opendss_auto(args.env_name, args.episodes, args.seed)
        print(f"  Reward:    {results['reward_mean']:.2f} +/- {results['reward_std']:.2f}")
        print(f"  Violation: {results['violation_mean']:.4f}")
        print(f"  Loss:      {results['loss_mean']:.4f}")
        print(f"  Time/step: {results['inference_time_per_step']*1000:.2f} ms")

    if args.method in ['opf', 'both']:
        print(f"\n--- Greedy Voltage Optimization (Greedy Heuristic, myopic) ---")
        results = evaluate_socp_opf(args.env_name, args.episodes, args.seed)
        print(f"  Reward:    {results['reward_mean']:.2f} +/- {results['reward_std']:.2f}")
        print(f"  Violation: {results['violation_mean']:.4f}")
        print(f"  Loss:      {results['loss_mean']:.4f}")
        print(f"  Time/step: {results['inference_time_per_step']*1000:.2f} ms")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()