"""
run_exp.py - Multi-Seed Experiment Runner
===================================================================
Fixed version with:
1. Proper heuristic baseline (not random actions)
2. Multi-seed experiments with statistical tests
3. Consistent evaluation across all methods

Usage:
    python run_exp.py --env 13Bus --seeds 5
    python run_exp.py --env all --seeds 5
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append('./powergym')

from powergym.env_register import make_env
from stable_baselines3 import PPO

# Import training functions
from train_marl import train_ippo, IPPO_Wrapper
from train_monolithic import train_monolithic

# Import fixed heuristic
from heuristic_agent import HeuristicController, GymnasiumCompatibilityWrapper


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 456, 789, 1011]

TIMESTEPS = {
    "13Bus": 50000,
    "34Bus": 100000,
    "123Bus": 200000
}

ENVIRONMENTS = ["13Bus", "34Bus", "123Bus"]


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(env_name: str, model_path: str, n_episodes: int = 10, 
                   agent_type: str = "monolithic", seed: int = None) -> dict:
    """
    Evaluate a trained DRL model.
    
    Args:
        env_name: Environment name
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        agent_type: "specialist" or "monolithic"
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create appropriate environment wrapper
    if agent_type == "specialist":
        env = IPPO_Wrapper(env_name)
    else:
        raw_env = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw_env)
    
    # Load model
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_violations = []
    episode_losses = []
    
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
    
    env.close()
    
    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_all": episode_rewards,
        "violation_mean": np.mean(episode_violations) if episode_violations else 0,
        "violation_std": np.std(episode_violations) if episode_violations else 0,
        "loss_mean": np.mean(episode_losses) if episode_losses else 0,
        "loss_std": np.std(episode_losses) if episode_losses else 0
    }


def evaluate_heuristic_fixed(env_name: str, n_episodes: int = 10, seed: int = None) -> dict:
    """
    Evaluate rule-based heuristic using PROPER Algorithm 1 logic.
    
    This is the FIXED version that implements actual voltage-based control
    instead of random actions.
    """
    raw_env = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw_env)
    
    # Use proper heuristic controller
    controller = HeuristicController(env_name)
    
    episode_rewards = []
    episode_violations = []
    episode_losses = []
    
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
            # FIXED: Use actual heuristic logic instead of random
            action = controller.get_action(obs, info)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            
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
    
    env.close()
    
    return {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "reward_all": episode_rewards,
        "violation_mean": np.mean(episode_violations) if episode_violations else 0,
        "violation_std": np.std(episode_violations) if episode_violations else 0,
        "loss_mean": np.mean(episode_losses) if episode_losses else 0,
        "loss_std": np.std(episode_losses) if episode_losses else 0
    }


# ============================================================
# MULTI-SEED EXPERIMENT RUNNER
# ============================================================

def run_multi_seed_experiments(env_name: str, seeds: list, output_dir: str,
                                skip_training: bool = False) -> dict:
    """
    Run complete multi-seed experiments for one environment.
    
    Returns structured results for all methods across all seeds.
    """
    print(f"\n{'#'*60}")
    print(f"# ENVIRONMENT: {env_name}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")
    
    results = {
        "specialist": {"training_data": [], "eval_results": []},
        "monolithic": {"training_data": [], "eval_results": []},
        "heuristic": {"eval_results": []}
    }
    
    timesteps = TIMESTEPS[env_name]
    
    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"SEED: {seed}")
        print(f"{'='*40}")
        
        # ===== 1. SPECIALIST ENSEMBLE =====
        if not skip_training:
            print(f"\n[1/3] Training Specialist Ensemble (seed={seed})...")
            model_name = f"{output_dir}/specialist_{env_name}_seed{seed}"
            try:
                _, training_data = train_ippo(
                    env_name=env_name,
                    steps=timesteps,
                    seed=seed,
                    model_name=model_name,
                    save_training_data=True,
                    verbose=0
                )
                results["specialist"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training specialist: {e}")
        
        # Evaluate specialist
        model_path = f"{output_dir}/specialist_{env_name}_seed{seed}"
        if os.path.exists(model_path + ".zip"):
            print(f"  Evaluating Specialist...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10, 
                                         agent_type="specialist", seed=seed)
            results["specialist"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f}")
        
        # ===== 2. MONOLITHIC AGENT =====
        if not skip_training:
            print(f"\n[2/3] Training Monolithic Agent (seed={seed})...")
            model_name = f"{output_dir}/monolithic_{env_name}_seed{seed}"
            try:
                _, training_data = train_monolithic(
                    env_name=env_name,
                    steps=timesteps,
                    seed=seed,
                    model_name=model_name,
                    save_training_data=True,
                    verbose=0
                )
                results["monolithic"]["training_data"].append(training_data)
            except Exception as e:
                print(f"  ERROR training monolithic: {e}")
        
        # Evaluate monolithic
        model_path = f"{output_dir}/monolithic_{env_name}_seed{seed}"
        if os.path.exists(model_path + ".zip"):
            print(f"  Evaluating Monolithic...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10,
                                         agent_type="monolithic", seed=seed)
            results["monolithic"]["eval_results"].append(eval_result)
            print(f"    Reward: {eval_result['reward_mean']:.2f}")
        
        # ===== 3. HEURISTIC BASELINE (FIXED) =====
        print(f"\n[3/3] Evaluating Heuristic Baseline (seed={seed})...")
        eval_result = evaluate_heuristic_fixed(env_name, n_episodes=10, seed=seed)
        results["heuristic"]["eval_results"].append(eval_result)
        print(f"    Reward: {eval_result['reward_mean']:.2f}")
    
    return results


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def compute_statistics(results: dict) -> dict:
    """
    Compute statistical measures across seeds.
    
    Returns aggregated statistics with confidence intervals and p-values.
    """
    stats_results = {}
    
    for method, data in results.items():
        eval_results = data.get("eval_results", [])
        if not eval_results:
            continue
        
        rewards = [r["reward_mean"] for r in eval_results]
        violations = [r["violation_mean"] for r in eval_results]
        losses = [r["loss_mean"] for r in eval_results]
        
        stats_results[method] = {
            "reward": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "ci_95": 1.96 * np.std(rewards) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0,
                "all": rewards
            },
            "violation": {
                "mean": np.mean(violations),
                "std": np.std(violations),
                "ci_95": 1.96 * np.std(violations) / np.sqrt(len(violations)) if len(violations) > 1 else 0,
                "all": violations
            },
            "loss": {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "all": losses
            },
            "n_seeds": len(rewards)
        }
    
    return stats_results


def perform_statistical_tests(stats_results: dict) -> dict:
    """
    Perform statistical significance tests between methods.
    
    Uses Welch's t-test for comparing means.
    """
    tests = {}
    
    methods = list(stats_results.keys())
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            comparison = f"{method1}_vs_{method2}"
            
            # Reward comparison
            rewards1 = stats_results[method1]["reward"]["all"]
            rewards2 = stats_results[method2]["reward"]["all"]
            
            if len(rewards1) > 1 and len(rewards2) > 1:
                t_stat, p_value = stats.ttest_ind(rewards1, rewards2, equal_var=False)
                tests[comparison] = {
                    "reward_p_value": p_value,
                    "reward_significant": p_value < 0.05,
                    "reward_t_stat": t_stat
                }
            
            # Violation comparison
            violations1 = stats_results[method1]["violation"]["all"]
            violations2 = stats_results[method2]["violation"]["all"]
            
            if len(violations1) > 1 and len(violations2) > 1:
                t_stat, p_value = stats.ttest_ind(violations1, violations2, equal_var=False)
                tests[comparison]["violation_p_value"] = p_value
                tests[comparison]["violation_significant"] = p_value < 0.05
    
    return tests

def plot_training_curves(all_results: dict, output_dir: str):
    """
    Generate publication-quality training curve plots.
    """
    for env_name, env_results in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {"specialist": "#2ecc71", "monolithic": "#3498db"}
        labels = {"specialist": "Specialist Ensemble (Ours)", "monolithic": "Monolithic PPO"}
        
        for method in ["specialist", "monolithic"]:
            if method not in env_results:
                continue
            
            training_data_list = env_results[method].get("training_data", [])
            if not training_data_list:
                continue
            
            # Aggregate rewards across seeds
            all_rewards = []
            max_len = 0
            
            for data in training_data_list:
                rewards = data.get("rewards", [])
                if rewards:
                    all_rewards.append(rewards)
                    max_len = max(max_len, len(rewards))
            
            if not all_rewards:
                continue
            
            # Pad to same length
            padded = []
            for r in all_rewards:
                if len(r) < max_len:
                    r = r + [r[-1]] * (max_len - len(r))  # Pad with last value
                padded.append(r[:max_len])
            
            rewards_array = np.array(padded)
            mean_rewards = np.mean(rewards_array, axis=0)
            std_rewards = np.std(rewards_array, axis=0)
            
            # Get timesteps
            timesteps = training_data_list[0].get("timesteps", list(range(len(mean_rewards))))
            if len(timesteps) != len(mean_rewards):
                timesteps = list(range(len(mean_rewards)))
            
            # Plot
            ax.plot(timesteps, mean_rewards, color=colors[method], 
                   label=labels[method], linewidth=2)
            ax.fill_between(timesteps, 
                          mean_rewards - std_rewards, 
                          mean_rewards + std_rewards,
                          color=colors[method], alpha=0.2)
        
        ax.set_xlabel('Training Timesteps', fontsize=12)
        ax.set_ylabel('Average Episode Reward', fontsize=12)
        ax.set_title(f'Training Convergence - IEEE {env_name} System', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"training_curve_{env_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

# ============================================================
# RESULT FORMATTING
# ============================================================

def create_results_table(all_results: dict) -> pd.DataFrame:
    """Create publication-ready results DataFrame."""
    rows = []
    
    for env_name, env_results in all_results.items():
        stats = compute_statistics(env_results)
        
        for method in ["specialist", "monolithic", "heuristic"]:
            if method not in stats:
                continue
            
            s = stats[method]
            rows.append({
                "System": env_name,
                "Method": method.capitalize(),
                "Reward (↑)": f"{s['reward']['mean']:.2f} ± {s['reward']['std']:.2f}",
                "Violation (↓)": f"{s['violation']['mean']:.4f} ± {s['violation']['std']:.4f}",
                "Loss (↓)": f"{s['loss']['mean']:.4f} ± {s['loss']['std']:.4f}",
                "N Seeds": s["n_seeds"],
                # Raw values for sorting
                "_reward_mean": s['reward']['mean'],
                "_violation_mean": s['violation']['mean'],
                "_loss_mean": s['loss']['mean']
            })
    
    return pd.DataFrame(rows)


def print_latex_table(df: pd.DataFrame, all_results: dict):
    """Print LaTeX-formatted table with statistical significance markers."""
    
    print("\n" + "="*80)
    print("LATEX TABLE (Copy to paper)")
    print("="*80)
    
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Performance Comparison (Mean $\pm$ Std, 5 Seeds)}")
    print(r"\begin{tabular}{llccccc}")
    print(r"\hline")
    print(r"\textbf{System} & \textbf{Method} & \textbf{Reward} $\uparrow$ & "
          r"\textbf{Viol.} $\downarrow$ & \textbf{Loss} $\downarrow$ & \textbf{p-value} \\")
    print(r"\hline")
    
    for env_name in ENVIRONMENTS:
        env_df = df[df["System"] == env_name]
        stats = compute_statistics(all_results.get(env_name, {}))
        tests = perform_statistical_tests(stats)
        
        for _, row in env_df.iterrows():
            method = row["Method"].lower()
            
            # Get p-value for this method vs specialist
            p_str = "-"
            if method != "specialist" and f"specialist_vs_{method}" in tests:
                p = tests[f"specialist_vs_{method}"].get("reward_p_value", 1.0)
                p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
                if p < 0.05:
                    p_str += "*"
            
            print(f"{row['System']} & {row['Method']} & {row['Reward (↑)']} & "
                  f"{row['Violation (↓)']} & {row['Loss (↓)']} & {p_str} \\\\")
        
        print(r"\hline")
    
    print(r"\end{tabular}")
    print(r"\label{tab:results}")
    print(r"\end{table*}")
    print("\n* indicates p < 0.05 (statistically significant)")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run Multi-Seed Experiments')
    parser.add_argument('--env', type=str, default='13Bus',
                        choices=['13Bus', '34Bus', '123Bus', 'all'])
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds (1-5)')
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
    print(f"# MULTI-SEED EXPERIMENTS")
    print(f"# Environments: {environments}")
    print(f"# Seeds: {seeds}")
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
    
    # Create results table
    df = create_results_table(all_results)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df[["System", "Method", "Reward (↑)", "Violation (↓)", "Loss (↓)", "N Seeds"]].to_string(index=False))
    
    # Save CSV
    df.to_csv(f"{output_dir}/results_summary.csv", index=False)
    
    # Print LaTeX table
    print_latex_table(df, all_results)

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_training_curves(all_results, output_dir)
    
    # Statistical tests summary
    print("\n" + "="*80)
    print("STATISTICAL TESTS (Specialist vs Others)")
    print("="*80)
    
    for env_name, results in all_results.items():
        stats = compute_statistics(results)
        tests = perform_statistical_tests(stats)
        
        print(f"\n{env_name}:")
        for comparison, test_results in tests.items():
            if "reward_p_value" in test_results:
                sig = "***" if test_results["reward_p_value"] < 0.001 else \
                      "**" if test_results["reward_p_value"] < 0.01 else \
                      "*" if test_results["reward_p_value"] < 0.05 else ""
                print(f"  {comparison}: p={test_results['reward_p_value']:.4f} {sig}")
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()