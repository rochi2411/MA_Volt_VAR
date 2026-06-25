"""
enhanced_statistics.py - Publication-Grade Statistical Validation
=================================================================
Drop-in replacements for compute_statistics() and perform_statistical_tests()
in run_exp.py, implementing best practices from:

  - Henderson et al. (AAAI 2018): "Deep RL That Matters" — seed sensitivity
  - Agarwal et al. (NeurIPS 2021): "Deep RL at the Edge of the Statistical
    Precipice" — IQM, bootstrap CIs, performance profiles

Usage:
    In run_exp.py, replace:
        from enhanced_statistics import (compute_statistics,
                                          perform_statistical_tests,
                                          plot_performance_profiles)

Features:
    - Bootstrap confidence intervals (BCa method)
    - Interquartile Mean (IQM) instead of plain mean
    - Mann-Whitney U test (non-parametric, valid for small N)
    - Effect size (Cohen's d)
    - Performance profiles for visual comparison
"""

import numpy as np
from scipy import stats
import warnings


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(data, statistic=np.mean, n_resamples=10000,
                 confidence=0.95, seed=None):
    """
    Compute bootstrap confidence interval using BCa method.
    Falls back to simple percentile CI if BCa produces invalid values.
    """
    data = np.array(data, dtype=float)
    n = len(data)
    
    if n < 2:
        val = statistic(data)
        return val, val
    
    # If all values are identical, CI is a point
    if np.std(data) < 1e-12:
        val = statistic(data)
        return val, val
    
    rng = np.random.RandomState(seed)
    boot_stats = np.zeros(n_resamples)
    
    for i in range(n_resamples):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    
    # Simple percentile CI as fallback
    alpha = (1 - confidence) / 2
    simple_lower = float(np.percentile(boot_stats, 100 * alpha))
    simple_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    
    # Try BCa correction
    try:
        observed = statistic(data)
        prop_below = np.mean(boot_stats < observed)
        
        # Guard: if proportion is 0 or 1, BCa can't compute z0
        if prop_below <= 0 or prop_below >= 1:
            return simple_lower, simple_upper
        
        z0 = stats.norm.ppf(prop_below)
        
        # Acceleration (jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stats[i] = statistic(jack_sample)
        jack_mean = np.mean(jackknife_stats)
        diff = jack_mean - jackknife_stats
        
        denom = 6 * (np.sum(diff ** 2)) ** 1.5
        if abs(denom) < 1e-15:
            return simple_lower, simple_upper
        a = np.sum(diff ** 3) / denom
        
        # Adjusted percentiles
        z_low = stats.norm.ppf(alpha)
        z_high = stats.norm.ppf(1 - alpha)
        
        denom_low = 1 - a * (z0 + z_low)
        denom_high = 1 - a * (z0 + z_high)
        
        if abs(denom_low) < 1e-10 or abs(denom_high) < 1e-10:
            return simple_lower, simple_upper
        
        a1 = stats.norm.cdf(z0 + (z0 + z_low) / denom_low)
        a2 = stats.norm.cdf(z0 + (z0 + z_high) / denom_high)
        
        # Validate
        if np.isnan(a1) or np.isnan(a2) or a1 < 0 or a1 > 1 or a2 < 0 or a2 > 1:
            return simple_lower, simple_upper
        
        a1 = np.clip(a1, 0.001, 0.999)
        a2 = np.clip(a2, 0.001, 0.999)
        
        lower = float(np.percentile(boot_stats, 100 * a1))
        upper = float(np.percentile(boot_stats, 100 * a2))
        return lower, upper
        
    except (ValueError, RuntimeWarning, FloatingPointError):
        return simple_lower, simple_upper


# ============================================================
# INTERQUARTILE MEAN (IQM)
# ============================================================

def interquartile_mean(data):
    """
    Compute Interquartile Mean — more robust than mean to outliers.
    Recommended by Agarwal et al. (NeurIPS 2021) for RL evaluation.
    
    Averages only the middle 50% of observations, discarding the
    bottom 25% and top 25%.
    """
    data = np.sort(np.array(data, dtype=float))
    n = len(data)
    if n < 4:
        return float(np.mean(data))
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    return float(np.mean(data[q1_idx:q3_idx]))


# ============================================================
# ENHANCED COMPUTE_STATISTICS (drop-in replacement)
# ============================================================

def compute_statistics(results: dict) -> dict:
    """
    Compute enhanced statistical measures across seeds.
    
    Drop-in replacement for run_exp.py's compute_statistics().
    Adds: IQM, bootstrap CIs, median.
    
    Returns:
        dict with per-method statistics
    """
    stats_results = {}
    
    for method, data in results.items():
        eval_results = data.get("eval_results", [])
        if not eval_results:
            continue
        
        rewards = [r["reward_mean"] for r in eval_results]
        violations = [r["violation_mean"] for r in eval_results]
        losses = [r["loss_mean"] for r in eval_results]
        
        # Bootstrap CIs for reward
        reward_ci = bootstrap_ci(rewards) if len(rewards) >= 2 else (np.mean(rewards), np.mean(rewards))
        
        stats_results[method] = {
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "median": float(np.median(rewards)),
                "iqm": interquartile_mean(rewards),
                "ci_95_lower": reward_ci[0],
                "ci_95_upper": reward_ci[1],
                "all": rewards
            },
            "violation": {
                "mean": float(np.mean(violations)),
                "std": float(np.std(violations)),
                "ci_95_lower": bootstrap_ci(violations)[0] if len(violations) >= 2 else float(np.mean(violations)),
                "ci_95_upper": bootstrap_ci(violations)[1] if len(violations) >= 2 else float(np.mean(violations)),
                "all": violations
            },
            "loss": {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "all": losses
            },
            "n_seeds": len(rewards)
        }
    
    return stats_results


# ============================================================
# ENHANCED STATISTICAL TESTS (drop-in replacement)
# ============================================================

def perform_statistical_tests(stats_results: dict) -> dict:
    """
    Perform non-parametric statistical significance tests.
    
    Drop-in replacement for run_exp.py's perform_statistical_tests().
    Uses Mann-Whitney U (valid for small N, no normality assumption)
    instead of Welch's t-test. Also reports Cohen's d effect size.
    """
    tests = {}
    methods = list(stats_results.keys())
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i + 1:]:
            comparison = f"{method1}_vs_{method2}"
            tests[comparison] = {}
            
            # --- Reward comparison ---
            r1 = stats_results[method1]["reward"]["all"]
            r2 = stats_results[method2]["reward"]["all"]
            
            if len(r1) >= 2 and len(r2) >= 2:
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, mw_p = stats.mannwhitneyu(
                        r1, r2, alternative='two-sided')
                except ValueError:
                    mw_p = 1.0
                    u_stat = 0
                
                # Also run Welch's t-test for comparison
                t_stat, t_p = stats.ttest_ind(r1, r2, equal_var=False)
                
                # Cohen's d effect size
                pooled_std = np.sqrt(
                    ((len(r1) - 1) * np.std(r1) ** 2 +
                     (len(r2) - 1) * np.std(r2) ** 2) /
                    (len(r1) + len(r2) - 2 + 1e-10)
                )
                cohens_d = (np.mean(r1) - np.mean(r2)) / (pooled_std + 1e-10)
                
                tests[comparison]["reward_p_value"] = float(mw_p)
                tests[comparison]["reward_p_value_ttest"] = float(t_p)
                tests[comparison]["reward_significant"] = mw_p < 0.05
                tests[comparison]["reward_u_stat"] = float(u_stat)
                tests[comparison]["reward_cohens_d"] = float(cohens_d)
                tests[comparison]["reward_effect_size"] = (
                    "large" if abs(cohens_d) > 0.8 else
                    "medium" if abs(cohens_d) > 0.5 else
                    "small" if abs(cohens_d) > 0.2 else "negligible"
                )
            
            # --- Violation comparison ---
            v1 = stats_results[method1]["violation"]["all"]
            v2 = stats_results[method2]["violation"]["all"]
            
            if len(v1) >= 2 and len(v2) >= 2:
                try:
                    _, vp = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                except ValueError:
                    vp = 1.0
                tests[comparison]["violation_p_value"] = float(vp)
                tests[comparison]["violation_significant"] = vp < 0.05
    
    return tests


# ============================================================
# PERFORMANCE PROFILE PLOTTING
# ============================================================

def compute_performance_profiles(all_results: dict, metric='reward',
                                  num_points=100):
    """
    Compute performance profiles across all environments.
    
    A performance profile shows P(score >= threshold) for each method
    across all seeds and environments. Recommended by Agarwal et al.
    
    Args:
        all_results: {env_name: {method: {eval_results: [...]}}}
        metric: 'reward' or 'violation'
        num_points: number of threshold points
    
    Returns:
        dict of {method: (thresholds, probabilities)}
    """
    # Collect all scores per method
    method_scores = {}
    
    for env_name, env_results in all_results.items():
        for method, data in env_results.items():
            eval_results = data.get("eval_results", [])
            if not eval_results:
                continue
            
            if method not in method_scores:
                method_scores[method] = []
            
            for r in eval_results:
                if metric == 'reward':
                    method_scores[method].append(r["reward_mean"])
                elif metric == 'violation':
                    # Negate so higher = better (fewer violations)
                    method_scores[method].append(-r["violation_mean"])
    
    if not method_scores:
        return {}
    
    # Normalize scores to [0, 1] across all methods
    all_scores = np.concatenate(list(method_scores.values()))
    score_min = np.min(all_scores)
    score_max = np.max(all_scores)
    score_range = score_max - score_min + 1e-10
    
    thresholds = np.linspace(0, 1, num_points)
    profiles = {}
    
    for method, scores in method_scores.items():
        normalized = (np.array(scores) - score_min) / score_range
        probs = np.array([np.mean(normalized >= t) for t in thresholds])
        profiles[method] = (thresholds, probs)
    
    return profiles


def plot_performance_profiles(all_results: dict, output_path: str,
                               metric='reward'):
    """
    Generate performance profile plot.
    
    Args:
        all_results: {env_name: {method: {eval_results: [...]}}}
        output_path: path to save the figure
        metric: 'reward' or 'violation'
    """
    try:
        import matplotlib.pyplot as plt
        from paper_style import apply_paper_style
        apply_paper_style()  # consistent sans-serif fonts + 300 DPI / TrueType export
    except ImportError:
        print("matplotlib not available, skipping performance profiles")
        return
    
    profiles = compute_performance_profiles(all_results, metric)
    if not profiles:
        return
    
    colors = {
        "specialist": "#2ecc71", "monolithic": "#3498db",
        "heuristic": "#e74c3c", "mappo": "#9b59b6",
        "maddpg": "#f39c12", "opendss_auto": "#1abc9c",
        "socp_opf": "#e67e22"
    }
    labels = {
        "specialist": "Specialist (Ours)", "monolithic": "Monolithic PPO",
        "heuristic": "Heuristic", "mappo": "MAPPO",
        "maddpg": "MADDPG", "opendss_auto": "OpenDSS Auto",
        "socp_opf": "SOCP-OPF"
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for method, (thresholds, probs) in profiles.items():
        color = colors.get(method, '#333333')
        label = labels.get(method, method)
        ax.plot(thresholds, probs, color=color, label=label, linewidth=2)
    
    ax.set_xlabel(f'Normalized {metric.capitalize()} Threshold (τ)', fontsize=12)
    ax.set_ylabel('Fraction of Runs with Score ≥ τ', fontsize=12)
    ax.set_title('Performance Profile (Agarwal et al. 2021)', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance profile: {output_path}")


# ============================================================
# SUMMARY PRINTER WITH ENHANCED STATISTICS
# ============================================================

def print_enhanced_summary(stats_results: dict, tests: dict,
                            env_name: str = ""):
    """Print a detailed statistical summary."""
    
    print(f"\n{'='*80}")
    print(f"ENHANCED STATISTICAL SUMMARY {f'- {env_name}' if env_name else ''}")
    print(f"{'='*80}")
    
    # Table header
    print(f"\n{'Method':<20} {'Mean±Std':<18} {'IQM':<10} "
          f"{'95% CI':<22} {'Violation':<12} {'N':<5}")
    print("-" * 87)
    
    for method, s in stats_results.items():
        r = s["reward"]
        v = s["violation"]
        ci_str = f"[{r['ci_95_lower']:.2f}, {r['ci_95_upper']:.2f}]"
        print(f"{method:<20} {r['mean']:>7.2f} ± {r['std']:<6.2f}  "
              f"{r['iqm']:>7.2f}   {ci_str:<22} {v['mean']:>8.4f}   "
              f"{s['n_seeds']}")
    
    # Pairwise tests
    if tests:
        print(f"\nPairwise Significance Tests (Mann-Whitney U):")
        print(f"{'Comparison':<35} {'p-value':<12} {'Effect':<12} {'Cohen d':<10}")
        print("-" * 69)
        for comp, t in tests.items():
            if "reward_p_value" in t:
                p = t["reward_p_value"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                eff = t.get("reward_effect_size", "?")
                d = t.get("reward_cohens_d", 0)
                print(f"{comp:<35} {p:<10.4f}{sig:<2} {eff:<12} {d:>8.2f}")


# ============================================================
# RECOMMENDED SEED LIST (10 seeds for stronger evidence)
# ============================================================

SEEDS_10 = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

