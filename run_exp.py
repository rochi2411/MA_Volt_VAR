"""
run_exp.py - Multi-Seed Experiment Runner (Revised for IEEE Transactions)
==========================================================================
Complete experiment pipeline with:
  1. Specialist Ensemble (IPPO with parameter sharing) - OURS
  2. Monolithic PPO baseline
  3. MAPPO baseline (centralized critic)
  4. MADDPG baseline (centralized critics, Gumbel-Softmax)
  5. Heuristic (rule-based) baseline
  6. OpenDSS Auto-Control + IEEE 1547 Droop baseline
  7. Model-based OPF (sensitivity-based) baseline

Systems: IEEE 13-Bus, 34-Bus, 123-Bus

Usage:
    python run_exp.py --env 13Bus --seeds 5
    python run_exp.py --env all --seeds 10
    python run_exp.py --env all --seeds 5 --skip_training
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from paper_style import apply_paper_style
apply_paper_style()  # consistent sans-serif fonts + 300 DPI / TrueType export

sys.path.append('./powergym')

from powergym.env_register import make_env
from stable_baselines3 import PPO

# Training functions
from train_marl import train_ippo, IPPO_Wrapper
from train_monolithic import train_monolithic
from train_mappo import train_mappo, MAPPO_Wrapper
from train_maddpg import train_maddpg, evaluate_maddpg

# Model-based baselines
from opendss_auto_baseline import evaluate_opendss_auto, evaluate_socp_opf
from opf_lindist3flow import evaluate_oracle_opf, evaluate_mpc_opf


def _opf_to_eval(r):
    """Adapt an opf_lindist3flow aggregate dict to run_exp's eval_result schema."""
    return {
        "reward_mean": r["reward_mean"],
        "reward_std": r["reward_std"],
        "violation_mean": r["violation_mean"],
        "loss_mean": r["loss_mean"],
        "reward_all": r.get("reward_all", []),
        "inference_time_per_step": r["solve_time_mean"] / 144.0,
        "pct_episodes_with_violation": r.get("pct_episodes_with_violation"),
        "cap_switches_mean": r.get("cap_switches_mean"),
        "reg_switches_mean": r.get("reg_switches_mean"),
    }

# Enhanced statistics
from enhanced_statistics import (
    compute_statistics,
    perform_statistical_tests,
    print_enhanced_summary,
    plot_performance_profiles
)

# Heuristic baseline
from heuristic_agent import HeuristicController, GymnasiumCompatibilityWrapper


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

TIMESTEPS = {
    "13Bus": 200000,     # 144-step episodes (24h at 10-min intervals)
    "34Bus": 400000,
    "123Bus": 800000,
}

ENVIRONMENTS = ["13Bus", "34Bus", "123Bus"]

# Methods that require training (vs. evaluate-only baselines)
TRAINED_METHODS = ["specialist", "monolithic", "mappo", "maddpg"]
# "Model-based OPF" is now the multi-period LinDist3Flow MILP oracle
# (opf_lindist3flow.py); "greedy_heuristic" is the old per-step greedy, kept
# as a weaker myopic reference row.
EVAL_ONLY_METHODS = ["heuristic", "opendss_auto", "greedy_heuristic",
                     "Model-based OPF", "OPF-MPC"]
ALL_METHODS = TRAINED_METHODS + EVAL_ONLY_METHODS

# Display names for tables and plots
METHOD_NAMES = {
    "specialist":   "Specialist (Ours)",
    "monolithic":   "Monolithic PPO",
    "mappo":        "MAPPO",
    "maddpg":       "MADDPG",
    "heuristic":    "Heuristic (Rule-Based)",
    "opendss_auto": "OpenDSS Auto + Droop",
    "greedy_heuristic":  "Greedy Heuristic (myopic)",
    "Model-based OPF":   "Model-based OPF (LinDist3Flow MILP, oracle)",
    "OPF-MPC":           "OPF-MPC (LinDist3Flow, persistence)",
}

METHOD_COLORS = {
    "specialist":   "#2ecc71",
    "monolithic":   "#3498db",
    "mappo":        "#9b59b6",
    "maddpg":       "#f39c12",
    "heuristic":    "#e74c3c",
    "opendss_auto": "#1abc9c",
    "greedy_heuristic":  "#95a5a6",
    "Model-based OPF":   "#e67e22",
    "OPF-MPC":           "#d35400",
}


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(env_name: str, model_path: str, n_episodes: int = 10,
                   agent_type: str = "monolithic", seed: int = None) -> dict:
    """Evaluate a trained SB3 model (specialist, monolithic, or mappo)."""
    if agent_type == "specialist":
        env = IPPO_Wrapper(env_name)
    elif agent_type == "mappo":
        env = MAPPO_Wrapper(env_name)
    else:
        raw_env = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw_env)

    model = PPO.load(model_path)

    episode_rewards = []
    episode_violations = []
    episode_losses = []

    total_steps = 0
    t_start = time.time()

    for ep in range(n_episodes):
        if seed is not None:
            obs, info = env.reset(seed=seed + ep)
        else:
            obs, info = env.reset()

        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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
    env.close()

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


def evaluate_heuristic_fixed(env_name: str, n_episodes: int = 10,
                              seed: int = None) -> dict:
    """Evaluate rule-based heuristic (Algorithm 1 in paper)."""
    raw_env = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw_env)
    controller = HeuristicController(env_name)

    episode_rewards = []
    episode_violations = []
    episode_losses = []

    total_steps = 0
    t_start = time.time()

    for ep in range(n_episodes):
        if seed is not None:
            obs, info = env.reset(seed=seed + ep)
        else:
            obs, info = env.reset()
        controller.reset()

        done = False
        ep_reward = 0
        ep_violations = []
        ep_losses = []

        while not done:
            action = controller.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
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
    env.close()

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


# ============================================================
# MULTI-SEED EXPERIMENT RUNNER
# ============================================================

def run_multi_seed_experiments(env_name: str, seeds: list, output_dir: str,
                               skip_training: bool = False) -> dict:
    """Run complete multi-seed experiments for one environment."""
    print(f"\n{'#'*60}")
    print(f"# ENVIRONMENT: {env_name}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")

    results = {m: {"training_data": [], "eval_results": []} for m in ALL_METHODS}

    timesteps = TIMESTEPS[env_name]

    # Compute number of agents for this environment.
    # Multi-agent wrappers (IPPO, MAPPO) consume N SB3 steps per env step
    # (one per agent). Scale their budget so they get equal env interactions.
    from powergym.ma_env import MultiAgentPowerGrid
    _tmp_ma = MultiAgentPowerGrid(env_name)
    n_agents = len(_tmp_ma.agents)
    ma_timesteps = timesteps * n_agents  # Fair budget for serialized MA methods
    print(f"  Agents: {n_agents} → MA methods get {ma_timesteps} SB3 steps "
          f"(= {timesteps} env steps)")
    del _tmp_ma
    n_methods = len(ALL_METHODS)

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"SEED: {seed}")
        print(f"{'='*40}")
        step_counter = 0

        # =============================================================
        # 1. SPECIALIST ENSEMBLE (Ours)
        # =============================================================
        step_counter += 1
        if not skip_training:
            print(f"\n[{step_counter}/{n_methods}] Training Specialist Ensemble (seed={seed})...")
            model_name = f"{output_dir}/specialist_{env_name}_seed{seed}"
            try:
                _, training_data = train_ippo(
                    env_name=env_name, steps=ma_timesteps, seed=seed,
                    model_name=model_name, save_training_data=True, verbose=0
                )
                results["specialist"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training specialist: {e}")
        else:
            # Load existing training data if available
            td_path = f"{output_dir}/specialist_{env_name}_seed{seed}_training_data.json"
            if os.path.exists(td_path):
                with open(td_path, 'r') as f:
                    results["specialist"]["training_data"].append(json.load(f))

        model_path = f"{output_dir}/specialist_{env_name}_seed{seed}"
        if os.path.exists(model_path + ".zip"):
            print(f"  Evaluating Specialist...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10,
                                         agent_type="specialist", seed=seed)
            results["specialist"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result['inference_time_per_step']*1000:.2f}ms")

        # =============================================================
        # 2. MONOLITHIC PPO
        # =============================================================
        step_counter += 1
        if not skip_training:
            print(f"\n[{step_counter}/{n_methods}] Training Monolithic PPO (seed={seed})...")
            model_name = f"{output_dir}/monolithic_{env_name}_seed{seed}"
            try:
                _, training_data = train_monolithic(
                    env_name=env_name, steps=timesteps, seed=seed,
                    model_name=model_name, save_training_data=True, verbose=0
                )
                results["monolithic"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training monolithic: {e}")
        else:
            td_path = f"{output_dir}/monolithic_{env_name}_seed{seed}_training_data.json"
            if os.path.exists(td_path):
                with open(td_path, 'r') as f:
                    results["monolithic"]["training_data"].append(json.load(f))

        model_path = f"{output_dir}/monolithic_{env_name}_seed{seed}"
        if os.path.exists(model_path + ".zip"):
            print(f"  Evaluating Monolithic...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10,
                                         agent_type="monolithic", seed=seed)
            results["monolithic"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result['inference_time_per_step']*1000:.2f}ms")

        # =============================================================
        # 3. MAPPO (Centralized Critic)
        # =============================================================
        step_counter += 1
        if not skip_training:
            print(f"\n[{step_counter}/{n_methods}] Training MAPPO (seed={seed})...")
            model_name = f"{output_dir}/mappo_{env_name}_seed{seed}"
            try:
                _, training_data = train_mappo(
                    env_name=env_name, steps=ma_timesteps, seed=seed,
                    model_name=model_name, save_training_data=True, verbose=0
                )
                results["mappo"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training MAPPO: {e}")
        else:
            td_path = f"{output_dir}/mappo_{env_name}_seed{seed}_training_data.json"
            if os.path.exists(td_path):
                with open(td_path, 'r') as f:
                    results["mappo"]["training_data"].append(json.load(f))

        model_path = f"{output_dir}/mappo_{env_name}_seed{seed}"
        if os.path.exists(model_path + ".zip"):
            print(f"  Evaluating MAPPO...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10,
                                         agent_type="mappo", seed=seed)
            results["mappo"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result['inference_time_per_step']*1000:.2f}ms")

        # =============================================================
        # 4. MADDPG (Centralized Critics, Gumbel-Softmax)
        # =============================================================
        step_counter += 1
        if not skip_training:
            print(f"\n[{step_counter}/{n_methods}] Training MADDPG (seed={seed})...")
            model_name = f"{output_dir}/maddpg_{env_name}_seed{seed}"
            try:
                _, training_data = train_maddpg(
                    env_name=env_name, steps=timesteps, seed=seed,
                    model_name=model_name, save_training_data=True, verbose=0
                )
                results["maddpg"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training MADDPG: {e}")
        else:
            td_path = f"{output_dir}/maddpg_{env_name}_seed{seed}_training_data.json"
            if os.path.exists(td_path):
                with open(td_path, 'r') as f:
                    results["maddpg"]["training_data"].append(json.load(f))

        model_path = f"{output_dir}/maddpg_{env_name}_seed{seed}.pt"
        if os.path.exists(model_path):
            print(f"  Evaluating MADDPG...")
            try:
                eval_result = evaluate_maddpg(env_name, model_path,
                                              n_episodes=10, seed=seed)
                results["maddpg"]["eval_results"].append(eval_result)
                print(f"    Reward: {eval_result['reward_mean']:.2f}")
            except Exception as e:
                print(f"  ERROR evaluating MADDPG: {e}")

        # =============================================================
        # 5. HEURISTIC (Rule-Based)
        # =============================================================
        step_counter += 1
        print(f"\n[{step_counter}/{n_methods}] Evaluating Heuristic (seed={seed})...")
        try:
            eval_result = evaluate_heuristic_fixed(env_name, n_episodes=10, seed=seed)
            results["heuristic"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result['inference_time_per_step']*1000:.2f}ms")
        except Exception as e:
            print(f"  ERROR in Heuristic: {e}")

        # =============================================================
        # 6. OPENDSS AUTO-CONTROL + IEEE 1547 DROOP
        # =============================================================
        step_counter += 1
        print(f"\n[{step_counter}/{n_methods}] Evaluating OpenDSS Auto-Control (seed={seed})...")
        try:
            eval_result = evaluate_opendss_auto(env_name, n_episodes=10, seed=seed)
            results["opendss_auto"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result.get('inference_time_per_step', 0)*1000:.2f}ms")
        except Exception as e:
            print(f"  ERROR in OpenDSS Auto: {e}")

        # =============================================================
        # 7. Greedy Heuristic (myopic) — old per-step controller, weak ref row
        # =============================================================
        step_counter += 1
        print(f"\n[{step_counter}/{n_methods}] Evaluating Greedy Heuristic (seed={seed})...")
        try:
            eval_result = evaluate_socp_opf(env_name, n_episodes=10, seed=seed)
            results["greedy_heuristic"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f} | "
                  f"Time/step: {eval_result.get('inference_time_per_step', 0)*1000:.2f}ms")
        except Exception as e:
            print(f"  ERROR in Greedy Heuristic: {e}")

        # =============================================================
        # 8. Model-based OPF: multi-period LinDist3Flow MILP (perfect-info
        #    oracle) + OPF-MPC (persistence). Deterministic optimizers, so
        #    evaluated once (on the first seed) over profiles 0-4.
        # =============================================================
        if len(results["Model-based OPF"]["eval_results"]) == 0:
            step_counter += 1
            print(f"\n[{step_counter}/{n_methods}] Evaluating Model-based OPF "
                  f"(LinDist3Flow MILP oracle)...")
            try:
                r = evaluate_oracle_opf(env_name, profiles=(0, 1, 2, 3, 4),
                                        horizon=144, tap_tier=1,
                                        time_limit=300, mip_gap=0.01)
                results["Model-based OPF"]["eval_results"].append(_opf_to_eval(r))
                print(f"    Reward: {r['reward_mean']:.2f} +/- {r['reward_std']:.2f} | "
                      f"solve {r['solve_time_mean']:.1f}s/episode")
            except Exception as e:
                print(f"  ERROR in Model-based OPF: {e}")
            try:
                rm = evaluate_mpc_opf(env_name, profiles=(0, 1, 2, 3, 4),
                                      horizon=144, window=6, tap_tier=1,
                                      time_limit=60, mip_gap=0.01)
                results["OPF-MPC"]["eval_results"].append(_opf_to_eval(rm))
                print(f"    OPF-MPC Reward: {rm['reward_mean']:.2f} +/- {rm['reward_std']:.2f}")
            except Exception as e:
                print(f"  ERROR in OPF-MPC: {e}")

        # Save after each seed (so partial results survive interruption)
        _save_path = f"{output_dir}/results_{env_name}.json"
        with open(_save_path, 'w') as _f:
            def _convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                return obj
            json.dump(results, _f, indent=2, default=_convert)
        print(f"\n  → Saved intermediate results ({len(results['specialist']['eval_results'])} seeds so far)")

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_training_curves(all_results: dict, output_dir: str):
    """Generate training curves plotted by EPISODE (not SB3 timesteps).
    Uses episode_rewards with rolling average for smooth curves.
    This ensures all methods are visually comparable since they all
    train for the same number of environment episodes."""
    window = 50  # Rolling average window

    for env_name, env_results in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for method in TRAINED_METHODS:
            if method not in env_results:
                continue

            training_data_list = env_results[method].get("training_data", [])
            if not training_data_list:
                continue

            # Collect per-episode rewards across seeds
            all_ep_rewards = []
            min_episodes = float('inf')
            for data in training_data_list:
                ep_rewards = data.get("episode_rewards", [])
                if ep_rewards:
                    all_ep_rewards.append(ep_rewards)
                    min_episodes = min(min_episodes, len(ep_rewards))

            if not all_ep_rewards or min_episodes < window:
                continue

            # Truncate all seeds to same length (shortest)
            truncated = [r[:int(min_episodes)] for r in all_ep_rewards]

            # Apply rolling average per seed
            smoothed = []
            for ep_rewards in truncated:
                arr = np.array(ep_rewards)
                kernel = np.ones(window) / window
                smooth = np.convolve(arr, kernel, mode='valid')
                smoothed.append(smooth)

            # Align lengths after convolution
            min_smooth = min(len(s) for s in smoothed)
            smoothed = np.array([s[:min_smooth] for s in smoothed])

            mean_r = np.mean(smoothed, axis=0)
            std_r = np.std(smoothed, axis=0)
            episodes = np.arange(window, window + len(mean_r))

            color = METHOD_COLORS.get(method, "#333333")
            label = f"{METHOD_NAMES.get(method, method)} (n={len(all_ep_rewards)})"

            ax.plot(episodes, mean_r, color=color, label=label, linewidth=2)
            ax.fill_between(episodes, mean_r - std_r, mean_r + std_r,
                            color=color, alpha=0.2)
            has_data = True

        if not has_data:
            plt.close()
            continue

        ax.set_xlabel('Training Episodes', fontsize=12)
        ax.set_ylabel('Episode Reward (rolling avg)', fontsize=12)
        ax.set_title(f'Training Convergence — IEEE {env_name}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"training_curve_{env_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        ax.set_xlabel('Training Timesteps', fontsize=12)
        ax.set_ylabel('Average Episode Reward', fontsize=12)
        ax.set_title(f'Training Convergence — IEEE {env_name} System', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"training_curve_{env_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


def plot_inference_time_comparison(all_results: dict, output_dir: str):
    """Bar chart comparing inference time per step across methods."""
    for env_name, env_results in all_results.items():
        methods_found = []
        times = []

        for method in ALL_METHODS:
            if method not in env_results:
                continue
            evals = env_results[method].get("eval_results", [])
            if not evals:
                continue
            avg_time = np.mean([
                e.get("inference_time_per_step", 0) for e in evals
            ]) * 1000  # ms
            if avg_time > 0:
                methods_found.append(method)
                times.append(avg_time)

        if not methods_found:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        display_names = [METHOD_NAMES.get(m, m) for m in methods_found]
        colors = [METHOD_COLORS.get(m, "#333") for m in methods_found]

        bars = ax.barh(display_names, times, color=colors)
        ax.set_xlabel('Inference Time per Step (ms)', fontsize=12)
        ax.set_title(f'Inference Latency — IEEE {env_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + max(times) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{t:.1f}ms', va='center', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"inference_time_{env_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# RESULT FORMATTING
# ============================================================

def create_results_table(all_results: dict) -> pd.DataFrame:
    """Create publication-ready results DataFrame."""
    rows = []

    for env_name, env_results in all_results.items():
        env_stats = compute_statistics(env_results)

        for method in ALL_METHODS:
            if method not in env_stats:
                continue

            s = env_stats[method]
            rows.append({
                "System": env_name,
                "Method": METHOD_NAMES.get(method, method),
                "Reward (↑)": f"{s['reward']['mean']:.2f} ± {s['reward']['std']:.2f}",
                "IQM": f"{s['reward']['iqm']:.2f}",
                "95% CI": f"[{s['reward']['ci_95_lower']:.2f}, {s['reward']['ci_95_upper']:.2f}]",
                "Violation (↓)": f"{s['violation']['mean']:.4f} ± {s['violation']['std']:.4f}",
                "Loss (↓)": f"{s['loss']['mean']:.4f}",
                "N Seeds": s["n_seeds"],
                "_reward_mean": s['reward']['mean'],
                "_violation_mean": s['violation']['mean'],
                "_loss_mean": s['loss']['mean'],
                "_method_key": method,
            })

    return pd.DataFrame(rows)


def print_latex_table(df: pd.DataFrame, all_results: dict):
    """Print LaTeX table with significance markers and bolded best values."""

    print("\n" + "=" * 80)
    print("LATEX TABLE (Copy to paper)")
    print("=" * 80)

    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Performance Comparison (Mean $\pm$ Std). "
          r"Statistical significance vs.\ Specialist: "
          r"$^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$}")
    print(r"\begin{tabular}{llccc}")
    print(r"\hline")
    print(r"\textbf{System} & \textbf{Method} & "
          r"\textbf{Reward} $\uparrow$ & "
          r"\textbf{Violation} $\downarrow$ & "
          r"\textbf{Loss} $\downarrow$ \\")
    print(r"\hline")

    for env_name in [e for e in ENVIRONMENTS if e in all_results]:
        env_stats = compute_statistics(all_results[env_name])
        env_tests = perform_statistical_tests(env_stats)

        available = [m for m in ALL_METHODS if m in env_stats]
        if not available:
            continue

        best_reward = max(env_stats[m]["reward"]["mean"] for m in available)
        best_violation = min(env_stats[m]["violation"]["mean"] for m in available)
        best_loss = min(env_stats[m]["loss"]["mean"] for m in available)

        first_row = True
        for method in ALL_METHODS:
            if method not in env_stats:
                continue

            s = env_stats[method]
            display_name = METHOD_NAMES.get(method, method)

            # Reward
            r_str = f"{s['reward']['mean']:.2f} $\\pm$ {s['reward']['std']:.2f}"
            if abs(s['reward']['mean'] - best_reward) < 1e-6:
                r_str = r"\textbf{" + r_str + "}"

            # Violation
            v_str = f"{s['violation']['mean']:.4f} $\\pm$ {s['violation']['std']:.4f}"
            if abs(s['violation']['mean'] - best_violation) < 1e-6:
                v_str = r"\textbf{" + v_str + "}"

            # Loss
            l_str = f"{s['loss']['mean']:.4f}"
            if abs(s['loss']['mean'] - best_loss) < 1e-6:
                l_str = r"\textbf{" + l_str + "}"

            # Significance vs specialist
            sig = ""
            if method != "specialist":
                p = None
                for key in [f"specialist_vs_{method}", f"{method}_vs_specialist"]:
                    if key in env_tests and "reward_p_value" in env_tests[key]:
                        p = env_tests[key]["reward_p_value"]
                        break
                if p is not None:
                    if p < 0.001:   sig = "$^{***}$"
                    elif p < 0.01:  sig = "$^{**}$"
                    elif p < 0.05:  sig = "$^{*}$"

            sys_str = (f"\\multirow{{{len(available)}}}{{*}}{{IEEE {env_name}}}"
                       if first_row else "")
            first_row = False

            print(f"{sys_str} & {display_name} & "
                  f"{r_str}{sig} & {v_str} & {l_str} \\\\")

        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\label{tab:results}")
    print(r"\end{table*}")


def print_computation_table(all_results: dict):
    """Print Table VII: Training time and inference latency comparison."""

    print("\n" + "=" * 80)
    print("COMPUTATION TIME TABLE (Table VII)")
    print("=" * 80)

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Computation Time Comparison}")
    print(r"\begin{tabular}{llcc}")
    print(r"\hline")
    print(r"\textbf{System} & \textbf{Method} & "
          r"\textbf{Train (s)} & \textbf{Inference (ms/step)} \\")
    print(r"\hline")

    for env_name in [e for e in ENVIRONMENTS if e in all_results]:
        env_results = all_results[env_name]
        first = True

        for method in ALL_METHODS:
            if method not in env_results:
                continue

            evals = env_results[method].get("eval_results", [])
            tdata = env_results[method].get("training_data", [])
            if not evals:
                continue

            # Training time
            train_times = [t.get("training_time_seconds", 0) for t in tdata]
            train_str = f"{np.mean(train_times):.0f}" if train_times else "---"

            # Inference time
            inf_times = [e.get("inference_time_per_step", 0) * 1000 for e in evals]
            inf_str = f"{np.mean(inf_times):.2f}" if any(t > 0 for t in inf_times) else "---"

            sys_str = env_name if first else ""
            first = False
            display = METHOD_NAMES.get(method, method)

            print(f"{sys_str} & {display} & {train_str} & {inf_str} \\\\")

        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\label{tab:computation}")
    print(r"\end{table}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Multi-Seed Experiments (IEEE Transactions Revision)')
    parser.add_argument('--env', type=str, default='13Bus',
                        choices=['13Bus', '34Bus', '123Bus', 'all'])
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds (1-10)')
    parser.add_argument('--output_dir', type=str, default='./experiment_results')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training, only evaluate existing models')

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir.lstrip('./'))
    os.makedirs(output_dir, exist_ok=True)

    seeds = SEEDS[:args.seeds]
    environments = ENVIRONMENTS if args.env == 'all' else [args.env]

    print(f"\n{'#'*60}")
    print(f"# MULTI-SEED EXPERIMENTS (IEEE Transactions Revision)")
    print(f"# Environments: {environments}")
    print(f"# Seeds ({len(seeds)}): {seeds}")
    print(f"# Methods: {ALL_METHODS}")
    print(f"# Output: {output_dir}")
    print(f"# Training: {'SKIP' if args.skip_training else 'ENABLED'}")
    print(f"{'#'*60}")

    all_results = {}

    for env_name in environments:
        results = run_multi_seed_experiments(
            env_name, seeds, output_dir, args.skip_training
        )
        all_results[env_name] = results

        # Save intermediate results
        with open(f"{output_dir}/results_{env_name}.json", 'w') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

    # ===== BACKFILL TRAINING DATA FROM INDIVIDUAL FILES =====
    # When --skip_training is used, training_data lists are empty, but
    # individual *_training_data.json files may exist from prior runs.
    method_prefixes = {
        "specialist": "specialist", "monolithic": "monolithic",
        "mappo": "mappo", "maddpg": "maddpg",
    }
    for env_name, env_results in all_results.items():
        for method, prefix in method_prefixes.items():
            if method not in env_results:
                continue
            tdata = env_results[method].get("training_data", [])
            if not tdata:
                loaded = []
                for fname in sorted(os.listdir(output_dir)):
                    if (fname.startswith(f"{prefix}_{env_name}_seed")
                            and fname.endswith("_training_data.json")):
                        with open(os.path.join(output_dir, fname), 'r') as f:
                            loaded.append(json.load(f))
                if loaded:
                    env_results[method]["training_data"] = loaded
                    print(f"  Backfilled {len(loaded)} training curves for "
                          f"{method}/{env_name}")

    # ===== RESULTS TABLE =====
    df = create_results_table(all_results)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    display_cols = ["System", "Method", "Reward (↑)", "IQM", "95% CI",
                    "Violation (↓)", "Loss (↓)", "N Seeds"]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))

    df.to_csv(f"{output_dir}/results_summary.csv", index=False)

    # ===== LATEX TABLES =====
    print_latex_table(df, all_results)
    print_computation_table(all_results)

    # ===== PLOTS =====
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plot_training_curves(all_results, output_dir)
    plot_inference_time_comparison(all_results, output_dir)

    # Performance profiles (only meaningful with 2+ environments)
    if len(environments) > 1:
        plot_performance_profiles(
            all_results,
            os.path.join(output_dir, "performance_profile_reward.png"),
            metric='reward'
        )
        plot_performance_profiles(
            all_results,
            os.path.join(output_dir, "performance_profile_violation.png"),
            metric='violation'
        )

    # ===== ENHANCED STATISTICS =====
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (Enhanced)")
    print("=" * 80)

    for env_name, results in all_results.items():
        env_stats = compute_statistics(results)
        env_tests = perform_statistical_tests(env_stats)
        print_enhanced_summary(env_stats, env_tests, env_name)

    # ===== DONE =====
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print(f"  - results_summary.csv")
    print(f"  - results_{{env}}.json (per environment)")
    print(f"  - training_curve_{{env}}.png")
    print(f"  - inference_time_{{env}}.png")
    if len(environments) > 1:
        print(f"  - performance_profile_reward.png")
        print(f"  - performance_profile_violation.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()