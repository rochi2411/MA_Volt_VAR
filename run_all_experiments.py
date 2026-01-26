"""
run_all_experiments.py - Complete Experiment Runner
====================================================
Runs all experiments needed for IEEE publication:
1. Multi-seed training for Specialist and Monolithic
2. Evaluation and metric collection
3. Results aggregation and plotting

Usage:
    python run_all_experiments.py                    # Run everything
    python run_all_experiments.py --env 13Bus        # Single environment
    python run_all_experiments.py --seeds 3          # Fewer seeds (faster)
    python run_all_experiments.py --eval_only        # Skip training, just evaluate
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

# Add powergym to path
sys.path.append('./powergym')

# Import training functions
from train_marl import train_ippo, IPPO_Wrapper  # Also import IPPO_Wrapper
from train_monolithic import train_monolithic
from powergym.env_register import make_env
from stable_baselines3 import PPO

# Gymnasium
import gymnasium as gym


# ============================================================
# GYMNASIUM COMPATIBILITY WRAPPER
# ============================================================

class GymnasiumCompatibilityWrapper(gym.Wrapper):
    """
    Wraps old-style Gym environments to be compatible with Gymnasium API.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
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
        
        return obs, info
    
    def step(self, action):
        result = self.env.step(action)
        
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        
        return obs, reward, terminated, truncated, info


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

OUTPUT_DIR = "./experiment_results"


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(env_name: str, model_path: str, n_episodes: int = 10, agent_type: str = "monolithic") -> dict:
    """
    Evaluate a trained model and return metrics.
    
    Args:
        env_name: Environment name
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        agent_type: "specialist" or "monolithic" - determines which env wrapper to use
    """
    # Use the same environment wrapper that was used during training
    if agent_type == "specialist":
        # Specialist was trained with IPPO_Wrapper
        env = IPPO_Wrapper(env_name)
    else:
        # Monolithic was trained with raw env + compatibility wrapper
        raw_env = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw_env)
    
    model = PPO.load(model_path)
    
    episode_rewards = []
    episode_violations = []
    episode_losses = []
    
    for ep in range(n_episodes):
        # Reset environment
        if agent_type == "specialist":
            obs, info = env.reset()
        else:
            obs, info = env.reset()
        
        done = False
        ep_reward = 0
        ep_viol_list = []
        ep_loss_list = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            if agent_type == "specialist":
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated if isinstance(terminated, bool) else (terminated or truncated)
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            ep_reward += reward
            
            # Extract metrics from info dict
            if isinstance(info, dict):
                # Voltage violation - use constraint_cost or vol_reward
                if 'constraint_cost' in info:
                    ep_viol_list.append(info['constraint_cost'])
                elif 'vol_reward' in info:
                    ep_viol_list.append(abs(info['vol_reward']))  # Use absolute value
                    
                # Power loss
                if 'power_loss_ratio' in info:
                    ep_loss_list.append(info['power_loss_ratio'])
        
        episode_rewards.append(ep_reward)
        if ep_viol_list:
            episode_violations.append(np.mean(ep_viol_list))
        if ep_loss_list:
            episode_losses.append(np.mean(ep_loss_list))
    
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


def evaluate_heuristic(env_name: str, n_episodes: int = 10) -> dict:
    """
    Evaluate rule-based heuristic baseline.
    """
    # Wrap environment for compatibility
    raw_env = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw_env)
    
    episode_rewards = []
    episode_violations = []
    episode_losses = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_viol_list = []
        ep_loss_list = []
        
        while not done:
            # Simple heuristic: sample random action
            # Replace with actual heuristic logic if available
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            
            # Extract metrics from info dict
            if isinstance(info, dict):
                # Voltage violation - use constraint_cost or vol_reward
                if 'constraint_cost' in info:
                    ep_viol_list.append(info['constraint_cost'])
                elif 'vol_reward' in info:
                    ep_viol_list.append(abs(info['vol_reward']))
                    
                # Power loss
                if 'power_loss_ratio' in info:
                    ep_loss_list.append(info['power_loss_ratio'])
        
        episode_rewards.append(ep_reward)
        if ep_viol_list:
            episode_violations.append(np.mean(ep_viol_list))
        if ep_loss_list:
            episode_losses.append(np.mean(ep_loss_list))
    
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
# TRAINING RUNNERS
# ============================================================

def run_training_experiments(env_name: str, seeds: list, output_dir: str):
    """
    Run all training experiments for one environment.
    """
    # Get absolute path to prevent directory change issues
    output_dir = os.path.abspath(output_dir)
    repo_root = os.path.abspath(os.path.dirname(__file__))
    
    results = {
        "env_name": env_name,
        "seeds": seeds,
        "specialist": {"training_data": [], "eval_results": []},
        "monolithic": {"training_data": [], "eval_results": []}
    }
    
    timesteps = TIMESTEPS[env_name]
    
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# {env_name} - Seed {seed}")
        print(f"{'#'*60}")
        
        # Always return to repo root before each training
        os.chdir(repo_root)
        
        # === Train Specialist ===
        print("\n[1/2] Training Specialist Ensemble...")
        try:
            model_name = os.path.join(output_dir, f"specialist_{env_name}_seed{seed}")
            model_path, training_data = train_ippo(
                env_name=env_name,
                steps=timesteps,
                seed=seed,
                model_name=model_name,
                verbose=1
            )
            
            # Return to repo root before evaluation
            os.chdir(repo_root)
            
            # Evaluate
            print("Evaluating Specialist...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10, agent_type="specialist")
            
            results["specialist"]["training_data"].append(training_data)
            results["specialist"]["eval_results"].append(eval_result)
            
            print(f"  Specialist Reward: {eval_result['reward_mean']:.2f} ± {eval_result['reward_std']:.2f}")
            
        except Exception as e:
            print(f"  ERROR in Specialist training: {e}")
            import traceback
            traceback.print_exc()
            results["specialist"]["eval_results"].append(None)
        
        # Return to repo root
        os.chdir(repo_root)
        
        # === Train Monolithic ===
        print("\n[2/2] Training Monolithic Agent...")
        try:
            model_name = os.path.join(output_dir, f"monolithic_{env_name}_seed{seed}")
            model_path, training_data = train_monolithic(
                env_name=env_name,
                steps=timesteps,
                seed=seed,
                model_name=model_name,
                verbose=1
            )
            
            # Return to repo root before evaluation
            os.chdir(repo_root)
            
            # Evaluate
            print("Evaluating Monolithic...")
            eval_result = evaluate_model(env_name, model_path, n_episodes=10, agent_type="monolithic")
            
            results["monolithic"]["training_data"].append(training_data)
            results["monolithic"]["eval_results"].append(eval_result)
            
            print(f"  Monolithic Reward: {eval_result['reward_mean']:.2f} ± {eval_result['reward_std']:.2f}")
            
        except Exception as e:
            print(f"  ERROR in Monolithic training: {e}")
            import traceback
            traceback.print_exc()
            results["monolithic"]["eval_results"].append(None)
        
        # Return to repo root
        os.chdir(repo_root)
    
    # === Evaluate Heuristic ===
    print("\nEvaluating Heuristic Baseline...")
    os.chdir(repo_root)
    try:
        heuristic_results = []
        for seed in seeds:
            np.random.seed(seed)
            os.chdir(repo_root)  # Reset directory each time
            eval_result = evaluate_heuristic(env_name, n_episodes=10)
            heuristic_results.append(eval_result)
            os.chdir(repo_root)
        results["heuristic"] = {"eval_results": heuristic_results}
        print(f"  Heuristic Reward: {np.mean([r['reward_mean'] for r in heuristic_results]):.2f}")
    except Exception as e:
        print(f"  ERROR in Heuristic evaluation: {e}")
        import traceback
        traceback.print_exc()
        results["heuristic"] = {"eval_results": []}
    
    return results


# ============================================================
# RESULTS AGGREGATION
# ============================================================

def aggregate_results(all_results: dict) -> pd.DataFrame:
    """
    Aggregate results into a publication-ready table.
    """
    rows = []
    
    for env_name, env_results in all_results.items():
        for method in ["heuristic", "monolithic", "specialist"]:
            if method not in env_results:
                continue
            
            eval_results = env_results[method].get("eval_results", [])
            eval_results = [r for r in eval_results if r is not None]
            
            if not eval_results:
                continue
            
            rewards = [r["reward_mean"] for r in eval_results]
            violations = [r["violation_mean"] for r in eval_results]
            losses = [r["loss_mean"] for r in eval_results]
            
            rows.append({
                "System": env_name,
                "Method": method.capitalize(),
                "Reward Mean": np.mean(rewards),
                "Reward Std": np.std(rewards),
                "Violation Mean": np.mean(violations),
                "Violation Std": np.std(violations),
                "Loss Mean": np.mean(losses),
                "Loss Std": np.std(losses),
                "N Seeds": len(eval_results)
            })
    
    return pd.DataFrame(rows)


def print_latex_table(df: pd.DataFrame):
    """
    Print results in LaTeX table format.
    """
    if df.empty:
        print("\nNo results to display - DataFrame is empty!")
        return
    
    print("\n" + "="*80)
    print("LATEX TABLE (Copy to paper)")
    print("="*80)
    
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Performance Comparison (Mean $\pm$ Std over 5 Seeds)}")
    print(r"\begin{tabular}{llccc}")
    print(r"\hline")
    print(r"\textbf{System} & \textbf{Method} & \textbf{Avg. Reward} & \textbf{Volt. Violation} & \textbf{Power Loss} \\")
    print(r"\hline")
    
    for env in ENVIRONMENTS:
        if "System" not in df.columns:
            continue
        env_df = df[df["System"] == env]
        for _, row in env_df.iterrows():
            reward_str = f"${row['Reward Mean']:.2f} \\pm {row['Reward Std']:.2f}$"
            violation_str = f"${row['Violation Mean']:.4f} \\pm {row['Violation Std']:.4f}$"
            loss_str = f"${row['Loss Mean']:.4f} \\pm {row['Loss Std']:.4f}$"
            
            print(f"{row['System']} & {row['Method']} & {reward_str} & {violation_str} & {loss_str} \\\\")
        print(r"\hline")
    
    print(r"\end{tabular}")
    print(r"\end{table*}")


# ============================================================
# PLOTTING
# ============================================================

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
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run All Experiments')
    parser.add_argument('--env', type=str, default='all',
                       choices=['13Bus', '34Bus', '123Bus', 'all'])
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds to use (max 5)')
    parser.add_argument('--output_dir', type=str, default='./experiment_results')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training, only evaluate existing models')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    seeds = SEEDS[:args.seeds]
    environments = ENVIRONMENTS if args.env == 'all' else [args.env]
    
    print(f"\n{'#'*60}")
    print(f"# MULTI-SEED EXPERIMENT SUITE")
    print(f"# Environments: {environments}")
    print(f"# Seeds: {seeds}")
    print(f"# Output: {args.output_dir}")
    print(f"{'#'*60}")
    
    all_results = {}
    
    for env_name in environments:
        if args.eval_only:
            print(f"\n[EVAL ONLY] Skipping training for {env_name}")
            # Load existing results if available
            results_file = os.path.join(args.output_dir, f"results_{env_name}.json")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    all_results[env_name] = json.load(f)
        else:
            # Run training
            results = run_training_experiments(env_name, seeds, args.output_dir)
            all_results[env_name] = results
            
            # Save intermediate results
            results_file = os.path.join(args.output_dir, f"results_{env_name}.json")
            with open(results_file, 'w') as f:
                # Make JSON serializable
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    return obj
                json.dump(results, f, indent=2, default=convert)
    
    # Aggregate and display results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    df = aggregate_results(all_results)
    print("\nResults DataFrame:")
    print(df.to_string(index=False))
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # Print LaTeX table
    print_latex_table(df)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_training_curves(all_results, args.output_dir)
    
    # Save all results
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()