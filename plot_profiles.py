"""
plot_profiles.py - Generate Load Profile and Solar Irradiance Diagrams
======================================================================
Produces two publication-quality figures:
  1. Load profiles (multiple episodes overlaid, showing diversity)
  2. Solar irradiance profile (single-day reference shape)

Usage:
    python plot_profiles.py                          # Default: 13Bus
    python plot_profiles.py --system 34Bus
    python plot_profiles.py --system all              # All three systems
    python plot_profiles.py --output_dir figures/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from paper_style import apply_paper_style
apply_paper_style()  # consistent sans-serif fonts + 300 DPI / TrueType export

# ============================================================
# CONFIGURATION
# ============================================================

SYSTEMS = ["13Bus", "34Bus", "123Bus"]
N_STEPS = 144
STEP_MINUTES = 10
HOURS = np.arange(N_STEPS) * STEP_MINUTES / 60.0  # 0 to ~24

# Representative loads to plot per system (high-load buses)
REPRESENTATIVE_LOADS = {
    "13Bus":  ["671", "634a", "675a", "652", "611"],
    "34Bus":  ["890", "844", "860", "848", "832"],  # will fallback if not found
    "123Bus": ["1", "48", "65", "76", "300"],
}

# Colors
LOAD_CMAP = plt.cm.tab10
SOLAR_COLOR = "#f39c12"
SOLAR_FILL = "#fdebd0"
EPISODE_ALPHA = 0.15
EPISODE_COLOR = "#3498db"


# ============================================================
# DATA LOADING
# ============================================================

def load_solar_profile(system_dir):
    """Load the single-day solar irradiance profile."""
    path = os.path.join(system_dir, "loadshape", "SolarShape_daily.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    return np.loadtxt(path)


def load_episode_profiles(system_dir, episode_idx=0):
    """Load all per-load profiles for a given episode."""
    ep_dir = os.path.join(system_dir, "loadshape", f"{episode_idx:03d}")
    if not os.path.exists(ep_dir):
        return {}
    profiles = {}
    for fname in sorted(os.listdir(ep_dir)):
        if fname.endswith(".csv") and fname != "solar_irradiance.csv":
            load_name = fname.replace(".csv", "")
            data = np.loadtxt(os.path.join(ep_dir, fname))
            if len(data) == N_STEPS:
                profiles[load_name] = data
    return profiles


def count_episodes(system_dir):
    """Count available episode folders."""
    ls_dir = os.path.join(system_dir, "loadshape")
    count = 0
    for d in os.listdir(ls_dir):
        if d.isdigit() and os.path.isdir(os.path.join(ls_dir, d)):
            count += 1
    return count


# ============================================================
# PLOTTING
# ============================================================

def plot_load_profiles(system_name, system_dir, output_dir):
    """
    Plot load profiles showing:
      - Top panel: representative loads for one episode (distinct colors)
      - Bottom panel: one load across multiple episodes (diversity)
    """
    n_episodes = count_episodes(system_dir)
    print(f"  {system_name}: {n_episodes} episodes available")

    # Load episode 0 for representative loads
    ep0 = load_episode_profiles(system_dir, 0)
    if not ep0:
        print(f"  WARNING: No load profiles found for {system_name}")
        return

    # Find representative loads that exist
    candidates = REPRESENTATIVE_LOADS.get(system_name, [])
    available = [l for l in candidates if l in ep0]
    if len(available) < 3:
        # Fallback: pick first 5 loads
        available = list(ep0.keys())[:5]
    rep_loads = available[:5]

    # Load same load across multiple episodes for diversity plot
    diversity_load = rep_loads[0]
    n_show = min(20, n_episodes)
    episode_indices = np.linspace(0, n_episodes - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    # --- Top panel: representative loads (single episode) ---
    ax1 = axes[0]
    for i, load_name in enumerate(rep_loads):
        color = LOAD_CMAP(i / max(len(rep_loads) - 1, 1))
        ax1.plot(HOURS, ep0[load_name], color=color, linewidth=1.5,
                 label=f"Load {load_name}", alpha=0.85)

    ax1.set_ylabel("Load Multiplier", fontsize=12)
    ax1.set_title(f"IEEE {system_name} — Representative Load Profiles (Episode 0)",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # --- Bottom panel: one load across episodes (diversity) ---
    ax2 = axes[1]
    for ep_idx in episode_indices:
        ep_data = load_episode_profiles(system_dir, ep_idx)
        if diversity_load in ep_data:
            ax2.plot(HOURS, ep_data[diversity_load],
                     color=EPISODE_COLOR, alpha=EPISODE_ALPHA, linewidth=0.8)

    # Overlay episode 0 prominently
    ax2.plot(HOURS, ep0[diversity_load], color="#e74c3c", linewidth=2.0,
             label=f"Episode 0", zorder=10)

    ax2.set_xlabel("Hour of Day", fontsize=12)
    ax2.set_ylabel("Load Multiplier", fontsize=12)
    ax2.set_title(f"Load \"{diversity_load}\" Across {n_show} Episodes "
                   f"(showing daily diversity)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0, 24)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.grid(True, alpha=0.3)

    # Save
    save_path = os.path.join(output_dir, f"load_profiles_{system_name}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    save_png = save_path.replace(".pdf", ".png")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_png}")


def plot_solar_irradiance(system_name, system_dir, output_dir):
    """
    Plot solar irradiance profile:
      - Filled area under curve
      - Annotated sunrise/sunset, peak
      - Night regions shaded
    """
    solar = load_solar_profile(system_dir)
    if solar is None:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    # Night shading
    ax.axvspan(0, 6, color="#2c3e50", alpha=0.08, label="Night")
    ax.axvspan(19, 24, color="#2c3e50", alpha=0.08)

    # Filled irradiance curve
    ax.fill_between(HOURS, solar, color=SOLAR_FILL, alpha=0.9)
    ax.plot(HOURS, solar, color=SOLAR_COLOR, linewidth=2.5, label="Solar Irradiance")

    # Peak annotation
    peak_idx = np.argmax(solar)
    peak_hour = HOURS[peak_idx]
    peak_val = solar[peak_idx]
    ax.annotate(f"Peak: {peak_val:.2f}\n({peak_hour:.1f}h)",
                xy=(peak_hour, peak_val),
                xytext=(peak_hour + 2.5, peak_val + 0.05),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#e67e22", alpha=0.9))

    # Sunrise/sunset markers
    sunrise_idx = np.argmax(solar > 0.01)
    sunset_idx = len(solar) - 1 - np.argmax(solar[::-1] > 0.01)
    sunrise_h = HOURS[sunrise_idx]
    sunset_h = HOURS[sunset_idx]

    ax.axvline(sunrise_h, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(sunset_h, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(sunrise_h + 0.15, 0.02, f"Sunrise\n{sunrise_h:.1f}h",
            fontsize=9, color="#e74c3c", va="bottom")
    ax.text(sunset_h - 0.15, 0.02, f"Sunset\n{sunset_h:.1f}h",
            fontsize=9, color="#e74c3c", va="bottom", ha="right")

    # Formatting
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Irradiance Multiplier", fontsize=12)
    ax.set_title(f"IEEE {system_name} — Solar Irradiance Profile "
                 f"(cloud factor = 0.15)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Save
    save_path = os.path.join(output_dir, f"solar_irradiance_{system_name}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    save_png = save_path.replace(".pdf", ".png")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_png}")


def plot_combined(system_name, system_dir, output_dir):
    """
    Combined figure: load profiles + solar irradiance on shared x-axis.
    Ideal for a single paper figure.
    """
    solar = load_solar_profile(system_dir)
    ep0 = load_episode_profiles(system_dir, 0)
    if not ep0 or solar is None:
        print(f"  Cannot create combined plot for {system_name}")
        return

    candidates = REPRESENTATIVE_LOADS.get(system_name, [])
    available = [l for l in candidates if l in ep0]
    if len(available) < 3:
        available = list(ep0.keys())[:5]
    rep_loads = available[:5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [1.2, 1],
                                                  "hspace": 0.08})

    # --- Top: Load profiles ---
    for i, load_name in enumerate(rep_loads):
        color = LOAD_CMAP(i / max(len(rep_loads) - 1, 1))
        ax1.plot(HOURS, ep0[load_name], color=color, linewidth=1.5,
                 label=f"{load_name}", alpha=0.85)

    ax1.set_ylabel("Load Multiplier", fontsize=12)
    ax1.set_title(f"IEEE {system_name} — Daily Load & Solar Profiles "
                   f"(10-min resolution, 144 steps)",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.9,
               title="Load Bus", title_fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.text(0.01, 0.95, "(a) Load Profiles", transform=ax1.transAxes,
             fontsize=11, fontweight="bold", va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- Bottom: Solar irradiance ---
    ax2.axvspan(0, 6, color="#2c3e50", alpha=0.06)
    ax2.axvspan(19, 24, color="#2c3e50", alpha=0.06)
    ax2.fill_between(HOURS, solar, color=SOLAR_FILL, alpha=0.9)
    ax2.plot(HOURS, solar, color=SOLAR_COLOR, linewidth=2.5)

    peak_idx = np.argmax(solar)
    ax2.annotate(f"Peak: {solar[peak_idx]:.2f}",
                 xy=(HOURS[peak_idx], solar[peak_idx]),
                 xytext=(HOURS[peak_idx] + 2, solar[peak_idx] - 0.1),
                 fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="#e67e22", alpha=0.9))

    ax2.set_xlabel("Hour of Day", fontsize=12)
    ax2.set_ylabel("Irradiance Multiplier", fontsize=12)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(-0.02, 1.05)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.grid(True, alpha=0.3)
    ax2.text(0.01, 0.95, "(b) Solar Irradiance", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Save
    save_path = os.path.join(output_dir, f"profiles_combined_{system_name}.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    save_png = save_path.replace(".pdf", ".png")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    print(f"  Saved: {save_png}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate load profile and solar irradiance diagrams")
    parser.add_argument("--system", type=str, default="13Bus",
                        choices=SYSTEMS + ["all"],
                        help="Which system to plot (default: 13Bus)")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for plots")
    parser.add_argument("--combined", action="store_true",
                        help="Also generate combined load+solar figure")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    systems = SYSTEMS if args.system == "all" else [args.system]

    for system_name in systems:
        system_dir = os.path.join(script_dir, "systems", system_name)
        if not os.path.exists(system_dir):
            print(f"WARNING: {system_dir} not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Generating plots for IEEE {system_name}")
        print(f"{'='*50}")

        plot_load_profiles(system_name, system_dir, output_dir)
        plot_solar_irradiance(system_name, system_dir, output_dir)

        if args.combined:
            plot_combined(system_name, system_dir, output_dir)

    print(f"\nDone! Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
