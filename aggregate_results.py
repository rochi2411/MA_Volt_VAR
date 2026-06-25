"""
aggregate_results.py - Combine Results from Separate Experiment Runs
=====================================================================
When experiments are run separately per environment (e.g., 13Bus first,
then 34Bus, then 123Bus), this script loads all saved JSON results and
produces the unified tables, plots, and LaTeX output.

Usage:
    python aggregate_results.py --results_dir ./experiment_results
    python aggregate_results.py --results_dir ./experiment_results --envs 13Bus 34Bus
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paper_style import apply_paper_style
apply_paper_style()  # consistent sans-serif fonts + 300 DPI / TrueType export

sys.path.append('./powergym')

from enhanced_statistics import (
    compute_statistics,
    perform_statistical_tests,
    print_enhanced_summary,
    plot_performance_profiles
)

# ============================================================
# CONFIGURATION (must match run_exp.py)
# ============================================================

ENVIRONMENTS = ["13Bus", "34Bus", "123Bus"]

ALL_METHODS = [
    "specialist", "monolithic", "mappo", "maddpg",
    "heuristic", "opendss_auto", "Model-based OPF"
]

METHOD_NAMES = {
    "specialist":   "Specialist (Ours)",
    "monolithic":   "Monolithic PPO",
    "mappo":        "MAPPO",
    "maddpg":       "MADDPG",
    "heuristic":    "Heuristic (Rule-Based)",
    "opendss_auto": "OpenDSS Auto + Droop",
    "Model-based OPF":     "Model-based OPF",
}

METHOD_COLORS = {
    "specialist":   "#2ecc71",
    "monolithic":   "#3498db",
    "mappo":        "#9b59b6",
    "maddpg":       "#f39c12",
    "heuristic":    "#e74c3c",
    "opendss_auto": "#2c3e50",
    "Model-based OPF":     "#8c564b",
}

TRAINED_METHODS = ["specialist", "monolithic", "mappo", "maddpg"]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_all_results(results_dir, envs=None):
    """Load per-environment JSON result files and backfill training data."""
    all_results = {}
    target_envs = envs or ENVIRONMENTS

    for env_name in target_envs:
        json_path = os.path.join(results_dir, f"results_{env_name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_results[env_name] = json.load(f)
            
            # Backfill training_data from individual JSON files if missing
            method_prefixes = {
                "specialist": "specialist",
                "monolithic": "monolithic",
                "mappo": "mappo",
                "maddpg": "maddpg",
            }
            for method, prefix in method_prefixes.items():
                if method not in all_results[env_name]:
                    continue
                tdata = all_results[env_name][method].get("training_data", [])
                if not tdata:
                    # Scan for individual training data files
                    loaded = []
                    for f in sorted(os.listdir(results_dir)):
                        if f.startswith(f"{prefix}_{env_name}_seed") and f.endswith("_training_data.json"):
                            td_path = os.path.join(results_dir, f)
                            with open(td_path, 'r') as fp:
                                loaded.append(json.load(fp))
                    if loaded:
                        all_results[env_name][method]["training_data"] = loaded
                        print(f"    Loaded {len(loaded)} training curves for {method}")

            n_methods = sum(
                1 for m in all_results[env_name]
                if all_results[env_name][m].get("eval_results")
            )
            n_seeds = max(
                (len(all_results[env_name][m].get("eval_results", []))
                 for m in all_results[env_name]),
                default=0
            )
            print(f"  Loaded {env_name}: {n_methods} methods, {n_seeds} seeds")
        else:
            print(f"  SKIP: {json_path} not found")

    return all_results


# ============================================================
# TABLES
# ============================================================

def create_results_table(all_results):
    """Create publication-ready DataFrame."""
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
                "_method_key": method,
            })
    return pd.DataFrame(rows)


def print_latex_table(all_results):
    """Print LaTeX table with significance markers and bolded best values."""

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
            display = METHOD_NAMES.get(method, method)

            r_str = f"{s['reward']['mean']:.2f} $\\pm$ {s['reward']['std']:.2f}"
            if abs(s['reward']['mean'] - best_reward) < 1e-6:
                r_str = r"\textbf{" + r_str + "}"

            v_str = f"{s['violation']['mean']:.4f} $\\pm$ {s['violation']['std']:.4f}"
            if abs(s['violation']['mean'] - best_violation) < 1e-6:
                v_str = r"\textbf{" + v_str + "}"

            l_str = f"{s['loss']['mean']:.4f}"
            if abs(s['loss']['mean'] - best_loss) < 1e-6:
                l_str = r"\textbf{" + l_str + "}"

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

            n_seeds = s['n_seeds']
            sys_str = (f"\\multirow{{{len(available)}}}{{*}}{{IEEE {env_name} ({n_seeds} seeds)}}"
                       if first_row else "")
            first_row = False

            print(f"{sys_str} & {display} & "
                  f"{r_str}{sig} & {v_str} & {l_str} \\\\")
        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\label{tab:results}")
    print(r"\end{table*}")


def print_computation_table(all_results):
    """Print training time and inference latency table."""

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

            train_times = [t.get("training_time_seconds", 0) for t in tdata]
            train_str = f"{np.mean(train_times):.0f}" if train_times else "---"

            inf_times = [e.get("inference_time_per_step", 0) * 1000 for e in evals]
            inf_str = f"{np.mean(inf_times):.2f}" if any(t > 0 for t in inf_times) else "---"

            sys_str = env_name if first else ""
            first = False
            print(f"{sys_str} & {METHOD_NAMES.get(method, method)} & "
                  f"{train_str} & {inf_str} \\\\")
        print(r"\hline")

    print(r"\end{tabular}")
    print(r"\label{tab:computation}")
    print(r"\end{table}")


# ============================================================
# PLOTS
# ============================================================

def plot_training_curves(all_results, output_dir):
    """Training curves plotted by episode with rolling average smoothing."""
    window = 50

    for env_name, env_results in all_results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for method in TRAINED_METHODS:
            if method not in env_results:
                continue
            tdata_list = env_results[method].get("training_data", [])
            if not tdata_list:
                continue

            all_ep_rewards = []
            min_episodes = float('inf')
            for data in tdata_list:
                ep_rewards = data.get("episode_rewards", [])
                if ep_rewards:
                    all_ep_rewards.append(ep_rewards)
                    min_episodes = min(min_episodes, len(ep_rewards))

            if not all_ep_rewards or min_episodes < window:
                continue

            truncated = [r[:int(min_episodes)] for r in all_ep_rewards]

            smoothed = []
            for ep_rewards in truncated:
                arr = np.array(ep_rewards)
                kernel = np.ones(window) / window
                smooth = np.convolve(arr, kernel, mode='valid')
                smoothed.append(smooth)

            min_smooth = min(len(s) for s in smoothed)
            smoothed = np.array([s[:min_smooth] for s in smoothed])

            mean_r = np.mean(smoothed, axis=0)
            std_r = np.std(smoothed, axis=0)
            episodes = np.arange(window, window + len(mean_r))

            color = METHOD_COLORS.get(method, "#333")
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

        path = os.path.join(output_dir, f"training_curve_{env_name}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_inference_comparison(all_results, output_dir):
    """Bar chart of inference time per step."""
    for env_name, env_results in all_results.items():
        methods_found = []
        times = []
        for method in ALL_METHODS:
            if method not in env_results:
                continue
            evals = env_results[method].get("eval_results", [])
            if not evals:
                continue
            avg = np.mean([e.get("inference_time_per_step", 0) for e in evals]) * 1000
            if avg > 0:
                methods_found.append(method)
                times.append(avg)

        if not methods_found:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        names = [METHOD_NAMES.get(m, m) for m in methods_found]
        colors = [METHOD_COLORS.get(m, "#333") for m in methods_found]
        bars = ax.barh(names, times, color=colors)
        # Log scale: latencies span ~2 orders of magnitude (sub-ms learned
        # policies vs. tens of ms for the OPF solve), so a linear axis would
        # crush the learned methods into invisible slivers.
        ax.set_xscale('log')
        ax.set_xlim(left=min(times) * 0.5, right=max(times) * 2.0)
        ax.set_xlabel('Inference Time per Step (ms, log scale)', fontsize=12)
        ax.set_title(f'Inference Latency — IEEE {env_name}', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x', which='both')

        for bar, t in zip(bars, times):
            ax.text(bar.get_width() * 1.05,
                    bar.get_y() + bar.get_height() / 2,
                    f'{t:.2f}ms', va='center', fontsize=10)

        plt.tight_layout()
        path = os.path.join(output_dir, f"inference_time_{env_name}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_reward_barplot(all_results, output_dir):
    """Grouped bar chart of mean reward across systems."""
    envs = [e for e in ENVIRONMENTS if e in all_results]
    if not envs:
        return

    methods_present = []
    for m in ALL_METHODS:
        if any(m in all_results[e] and all_results[e][m].get("eval_results")
               for e in envs):
            methods_present.append(m)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(envs))
    width = 0.8 / len(methods_present)

    for i, method in enumerate(methods_present):
        means = []
        stds = []
        for env in envs:
            stats = compute_statistics(all_results[env])
            if method in stats:
                means.append(stats[method]["reward"]["mean"])
                stds.append(stats[method]["reward"]["std"])
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(methods_present) / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "#333")
        label = METHOD_NAMES.get(method, method)
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=3)

    ax.set_xlabel('IEEE Test System', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Reward Comparison Across Systems', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'IEEE {e}' for e in envs])
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(output_dir, "reward_comparison_all_systems.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate experiment results from separate runs')
    parser.add_argument('--results_dir', type=str, default='./experiment_results')
    parser.add_argument('--envs', nargs='+', default=None,
                        help='Environments to include (default: all found)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir for plots (default: same as results_dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Aggregating Experiment Results")
    print(f"Results dir: {args.results_dir}")
    print(f"{'='*60}\n")

    # Load
    all_results = load_all_results(args.results_dir, args.envs)

    if not all_results:
        print("No results found!")
        return

    # Console table
    df = create_results_table(all_results)
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    display_cols = ["System", "Method", "Reward (↑)", "IQM", "95% CI",
                    "Violation (↓)", "Loss (↓)", "N Seeds"]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))

    df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)

    # LaTeX
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print(f"{'='*80}")
    print_latex_table(all_results)

    print(f"\n{'='*80}")
    print("COMPUTATION TIME TABLE")
    print(f"{'='*80}")
    print_computation_table(all_results)

    # Plots
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    plot_training_curves(all_results, output_dir)
    plot_inference_comparison(all_results, output_dir)
    plot_reward_barplot(all_results, output_dir)

    # Performance profiles (2+ environments)
    envs_found = list(all_results.keys())
    if len(envs_found) > 1:
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

    # Enhanced statistics
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    for env_name, results in all_results.items():
        env_stats = compute_statistics(results)
        env_tests = perform_statistical_tests(env_stats)
        print_enhanced_summary(env_stats, env_tests, env_name)

    # Done
    print(f"\n{'='*80}")
    print("AGGREGATION COMPLETE!")
    print(f"Environments: {list(all_results.keys())}")
    print(f"Output: {output_dir}/")
    print(f"  - results_summary.csv")
    print(f"  - training_curve_{{env}}.png")
    print(f"  - inference_time_{{env}}.png")
    print(f"  - reward_comparison_all_systems.png")
    if len(envs_found) > 1:
        print(f"  - performance_profile_reward.png")
        print(f"  - performance_profile_violation.png")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()