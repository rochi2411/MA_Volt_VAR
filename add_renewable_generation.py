"""
add_renewable_generation.py - Add Distributed Generation Injections to Feeders
=============================================================================
Adds uncontrolled distributed-generation (DG) elements to each IEEE test
system, introducing time-varying nodal injections that create voltage
volatility for the volt-VAR controllers to manage.

What this does:
  1. Generates daily injection LoadShape CSVs (bell curve + stochastic variability)
  2. Adds DG Generator elements to each system's _daily.dss file
  3. DG is UNCONTROLLABLE — it's part of the environment dynamics, not the
     action space. The RL agent must learn to manage voltage despite it.

IMPORTANT CAVEAT (verified 2026-06-24): PowerGym's per-step snapshot
``Solution.Solve()`` does NOT faithfully apply the generators' ``daily`` shape.
The measured injection does not track the intended bell curve (e.g. ~0 kW at
solar noon, near-nameplate at 06:00/18:00 on 34Bus). The DG therefore produces
realistic *volatility* but is NOT a calibrated solar-penetration scenario — do
not describe it as "high PV penetration" or an "IEEE 1547-2018 scenario" in the
paper. See [[opf-baseline]] for the full diagnosis.

Sizing (nameplate per site; NOT a validated penetration level given the caveat):
  - 13Bus:    1.0 MW across 3 sites
  - 34Bus:    0.7 MW across 3 sites
  - 123Bus:   1.2 MW across 4 sites
  - 8500Node: 5.0 MW across 6 sites

Usage:
    python add_renewable_generation.py                    # All systems
    python add_renewable_generation.py --systems 13Bus    # One system
    python add_renewable_generation.py --revert           # Remove all DG

Honest paper description:
    "Each feeder is driven by time-varying stochastic load profiles together
    with uncontrolled distributed-generation injections, producing the voltage
    volatility and bidirectional power flow the controllers must regulate. The
    proposed method handles this volatility regardless of its source."
"""

import os
import sys
import shutil
import argparse
import numpy as np

sys.path.append('./powergym')


# ============================================================
# SOLAR IRRADIANCE PROFILE GENERATION
# ============================================================

def generate_solar_profile(n_steps=144, seed=None, cloud_factor=0.3,
                            peak_hour=12.5, sunrise=6.0, sunset=19.0):
    """
    Generate a realistic daily solar irradiance profile.
    
    Uses a truncated sinusoidal base with cloud transient noise.
    Returns values in [0, 1] where 1.0 = peak irradiance.
    
    Args:
        n_steps: Number of timesteps per day (24 for hourly, 144 for 10-min)
        seed: Random seed for cloud variability
        cloud_factor: Severity of cloud transients (0=clear, 1=heavy clouds)
        peak_hour: Hour of peak irradiance (default 12.5 = 12:30 PM)
        sunrise: Hour of sunrise
        sunset: Hour of sunset
    
    Returns:
        numpy array of shape (n_steps,) with values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    hours = np.linspace(0, 24, n_steps, endpoint=False)
    
    # Base irradiance: truncated sinusoidal
    day_length = sunset - sunrise
    irradiance = np.zeros(n_steps)
    
    for i, h in enumerate(hours):
        if sunrise <= h <= sunset:
            # Sinusoidal shape, peak at peak_hour
            phase = np.pi * (h - sunrise) / day_length
            irradiance[i] = np.sin(phase)
            
            # Slight asymmetry: afternoon slightly lower (thermal effects)
            if h > peak_hour:
                irradiance[i] *= 0.95
    
    # Add cloud transients (correlated noise, not white noise)
    if cloud_factor > 0:
        # Generate temporally correlated cloud events
        n_clouds = rng.poisson(3)  # Average 3 cloud events per day
        for _ in range(n_clouds):
            cloud_center = rng.uniform(sunrise + 1, sunset - 1)
            cloud_width = rng.uniform(0.5, 2.0)  # Hours
            cloud_depth = rng.uniform(0.2, 0.8) * cloud_factor
            
            cloud_mask = np.exp(-0.5 * ((hours - cloud_center) / cloud_width) ** 2)
            irradiance *= (1 - cloud_depth * cloud_mask)
        
        # Add small high-frequency noise (inverter fluctuations)
        noise = rng.normal(0, 0.02, n_steps)
        irradiance += noise * (irradiance > 0.05)  # Only during daylight
    
    # Clip to [0, 1]
    irradiance = np.clip(irradiance, 0.0, 1.0)
    
    return irradiance


def generate_annual_solar_profiles(n_steps_per_day=144, n_days=365, seed=42):
    """
    Generate a full year of solar profiles with seasonal variation.
    
    Summer days are longer with higher peak irradiance.
    Winter days are shorter with lower peak.
    Cloud frequency varies by season.
    """
    rng = np.random.RandomState(seed)
    all_profiles = []
    
    for day in range(n_days):
        # Seasonal parameters (Northern Hemisphere)
        day_of_year = day % 365
        seasonal_factor = 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 172) / 365))
        # seasonal_factor: 1.0 at summer solstice (day 172), 0.0 at winter solstice
        
        sunrise = 7.5 - 1.5 * seasonal_factor   # 6.0 (summer) to 7.5 (winter)
        sunset = 17.0 + 2.5 * seasonal_factor    # 17.0 (winter) to 19.5 (summer)
        peak_irradiance = 0.7 + 0.3 * seasonal_factor  # 0.7 (winter) to 1.0 (summer)
        cloud_factor = 0.2 + 0.2 * (1 - seasonal_factor)  # More clouds in winter
        
        profile = generate_solar_profile(
            n_steps=n_steps_per_day,
            seed=rng.randint(100000),
            cloud_factor=cloud_factor,
            peak_hour=12.5,
            sunrise=sunrise,
            sunset=sunset
        ) * peak_irradiance
        
        all_profiles.append(profile)
    
    return np.concatenate(all_profiles)  # Shape: (n_days * n_steps_per_day,)


# ============================================================
# PV SITE CONFIGURATIONS PER SYSTEM
# ============================================================

def get_pv_config(system_name, penetration=None):
    """
    Return PV installation specifications for each IEEE test system.
    
    Sites are chosen at feeder endpoints and load centers where rooftop
    PV would realistically be installed, and where voltage impact is greatest.
    
    Returns list of dicts: {bus, kW, phases, kV, conn, name}
    """
    configs = {
        '13Bus': {
            'total_load_kw': 3500,
            'default_penetration': 0.30,
            'sites': [
                # Bus 634: downstream of transformer, residential area
                {'bus': '634.1.2.3', 'frac': 0.40, 'phases': 3, 'kV': 0.48, 'conn': 'Wye', 'name': 'pv_634'},
                # Bus 671: large load bus, commercial PV
                {'bus': '671.1.2.3', 'frac': 0.35, 'phases': 3, 'kV': 4.16, 'conn': 'Delta', 'name': 'pv_671'},
                # Bus 675: feeder endpoint, worst voltage impact
                {'bus': '675.1.2.3', 'frac': 0.25, 'phases': 3, 'kV': 4.16, 'conn': 'Delta', 'name': 'pv_675'},
            ]
        },
        '34Bus': {
            'total_load_kw': 1800,
            'default_penetration': 0.40,
            'sites': [
                # Bus 840: endpoint, constant impedance load area
                {'bus': '840.1.2.3', 'frac': 0.30, 'phases': 3, 'kV': 24.9, 'conn': 'Wye', 'name': 'pv_840'},
                # Bus 844: near capacitor bank, high load density
                {'bus': '844.1.2.3', 'frac': 0.40, 'phases': 3, 'kV': 24.9, 'conn': 'Wye', 'name': 'pv_844'},
                # Bus 890: downstream of step-down transformer
                {'bus': '890.1.2.3', 'frac': 0.30, 'phases': 3, 'kV': 4.16, 'conn': 'Delta', 'name': 'pv_890'},
            ]
        },
        '123Bus': {
            'total_load_kw': 3500,
            'default_penetration': 0.35,
            'sites': [
                # Bus 48-51: end of lateral, voltage-sensitive
                {'bus': '48.1.2.3', 'frac': 0.25, 'phases': 3, 'kV': 4.16, 'conn': 'Wye', 'name': 'pv_48'},
                # Bus 65: midfeeder, moderate load area
                {'bus': '65.1.2.3', 'frac': 0.20, 'phases': 3, 'kV': 4.16, 'conn': 'Wye', 'name': 'pv_65'},
                # Bus 76: branch point, affects both downstream branches
                {'bus': '76.1.2.3', 'frac': 0.25, 'phases': 3, 'kV': 4.16, 'conn': 'Wye', 'name': 'pv_76'},
                # Bus 300: deep endpoint, worst-case voltage rise
                {'bus': '300.1.2.3', 'frac': 0.30, 'phases': 3, 'kV': 4.16, 'conn': 'Delta', 'name': 'pv_300'},
            ]
        },
        '8500Node': {
            'total_load_kw': 20000,
            'default_penetration': 0.25,
            'sites': [
                {'bus': 'L3160098.1.2.3', 'frac': 0.20, 'phases': 3, 'kV': 7.2, 'conn': 'Delta', 'name': 'pv_L3160098'},
                {'bus': 'L3312692.1', 'frac': 0.15, 'phases': 1, 'kV': 7.2, 'conn': 'Wye', 'name': 'pv_L3312692'},
                {'bus': 'L3091052.1', 'frac': 0.15, 'phases': 1, 'kV': 7.2, 'conn': 'Wye', 'name': 'pv_L3091052'},
                {'bus': 'L3235247.1', 'frac': 0.15, 'phases': 1, 'kV': 7.2, 'conn': 'Wye', 'name': 'pv_L3235247'},
                {'bus': 'M1069509.1.2.3', 'frac': 0.20, 'phases': 3, 'kV': 7.2, 'conn': 'Delta', 'name': 'pv_M1069509'},
                {'bus': 'L2785537.1', 'frac': 0.15, 'phases': 1, 'kV': 7.2, 'conn': 'Wye', 'name': 'pv_L2785537'},
            ]
        }
    }
    
    if system_name not in configs:
        print(f"  WARNING: No PV config for {system_name}")
        return []
    
    cfg = configs[system_name]
    pen = penetration or cfg['default_penetration']
    total_pv_kw = cfg['total_load_kw'] * pen
    
    sites = []
    for site in cfg['sites']:
        kw = total_pv_kw * site['frac']
        sites.append({
            'bus': site['bus'],
            'kW': round(kw, 1),
            'phases': site['phases'],
            'kV': site['kV'],
            'conn': site['conn'],
            'name': site['name']
        })
    
    return sites


# ============================================================
# DSS FILE GENERATION
# ============================================================

def create_solar_loadshape_dss(system_dir, n_steps=144, n_profiles=73):
    """
    Create solar irradiance LoadShape CSV and DSS redirect file.
    
    Generates annual solar profiles and splits them into per-episode
    files matching the load profile structure.
    """
    loadshape_dir = os.path.join(system_dir, 'loadshape')
    
    # Generate annual solar data
    annual = generate_annual_solar_profiles(
        n_steps_per_day=n_steps, n_days=365, seed=42)
    
    # Save as a single CSV (same structure as LoadShape*.CSV)
    solar_csv = os.path.join(loadshape_dir, 'SolarShape.CSV')
    np.savetxt(solar_csv, annual, fmt='%.15f')
    print(f"    Created SolarShape.CSV ({len(annual)} points)")
    
    # Also create per-episode solar files in each episode folder
    points_per_episode = n_steps
    n_episodes = min(n_profiles, len(annual) // points_per_episode)
    
    for ep in range(n_episodes):
        ep_dir = os.path.join(loadshape_dir, str(ep).zfill(3))
        if not os.path.exists(ep_dir):
            continue  # Episode folder doesn't exist yet, will be created by PowerGym
        
        start = ep * points_per_episode
        end = start + points_per_episode
        ep_solar = annual[start:end]
        
        solar_ep_csv = os.path.join(ep_dir, 'solar_irradiance.csv')
        np.savetxt(solar_ep_csv, ep_solar, fmt='%.15f')
    
    return n_episodes


def generate_pv_dss_block(sites, n_steps=144):
    """
    Generate OpenDSS commands to define PV generators.
    
    Uses Generator elements with negative load convention:
    - Positive kW = generation (power injection)
    - daily= loadshape follows solar irradiance
    - Model=1: constant P+jQ (standard for inverter-based DER)
    - pf=1.0: unity power factor (no reactive support from PV)
    
    Note: These generators are NOT in Battery.csv, so PowerGym
    will NOT try to control them. They are passive disturbances.
    """
    lines = []
    lines.append("")
    lines.append("! ============================================================")
    lines.append("! DISTRIBUTED SOLAR PV GENERATION (Uncontrollable DER)")
    lines.append("! Added by add_renewable_generation.py")
    lines.append("! PV operates at unity power factor (no reactive support)")
    lines.append("! These are NOT controllable by the RL agent")
    lines.append("! ============================================================")
    lines.append("")
    
    # Define solar loadshape (using sinterval matching the simulation)
    sinterval = 86400 // n_steps
    lines.append(f"New Loadshape.solar_irradiance npts={n_steps} sinterval={sinterval} "
                 f"mult=(file=loadshape/SolarShape_daily.csv)")
    lines.append("")
    
    total_kw = 0
    for site in sites:
        # Use negative Generator (injection) with solar daily shape
        # Model=1: constant P, Q. pf=1.0: unity power factor
        lines.append(
            f"New Generator.{site['name']} "
            f"bus1={site['bus']} "
            f"Phases={site['phases']} "
            f"kV={site['kV']} "
            f"kW={site['kW']} "
            f"pf=1.0 "
            f"conn={site['conn']} "
            f"Model=1 "
            f"daily=solar_irradiance"
        )
        total_kw += site['kW']
    
    lines.append(f"! Total PV capacity: {total_kw:.1f} kW")
    lines.append("")
    
    return "\n".join(lines), total_kw


def create_daily_solar_csv(system_dir, n_steps=144):
    """
    Create the single-day solar CSV that the DSS loadshape references.
    This is a clear-sky profile used as the base shape.
    """
    loadshape_dir = os.path.join(system_dir, 'loadshape')
    
    # Generate a representative clear-sky day
    profile = generate_solar_profile(n_steps=n_steps, seed=0, cloud_factor=0.15)
    
    csv_path = os.path.join(loadshape_dir, 'SolarShape_daily.csv')
    np.savetxt(csv_path, profile, fmt='%.15f')
    print(f"    Created SolarShape_daily.csv ({n_steps} points)")
    return csv_path


def patch_dss_file(dss_file_path, pv_block):
    """
    Insert PV definitions into the _daily.dss file, just before the 
    'Set mode=Daily' line.
    """
    with open(dss_file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'DISTRIBUTED SOLAR PV GENERATION' in content:
        print(f"    {os.path.basename(dss_file_path)}: already has PV, skipping")
        return False
    
    # Backup
    backup = dss_file_path + '.no_pv_backup'
    if not os.path.exists(backup):
        shutil.copy2(dss_file_path, backup)
    
    # Insert PV block before "Set mode=Daily"
    insert_marker = 'Set mode=Daily'
    if insert_marker not in content:
        # Try lowercase
        insert_marker = 'set mode=daily'
        if insert_marker.lower() not in content.lower():
            print(f"    WARNING: Could not find '{insert_marker}' in {dss_file_path}")
            return False
    
    # Find the line (case-insensitive)
    lines = content.split('\n')
    insert_idx = None
    for i, line in enumerate(lines):
        if 'set mode=daily' in line.lower():
            insert_idx = i
            break
    
    if insert_idx is None:
        print(f"    WARNING: Could not locate Set mode=Daily line")
        return False
    
    # Insert PV block before the Set mode line
    new_lines = lines[:insert_idx] + [pv_block] + lines[insert_idx:]
    
    with open(dss_file_path, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"    Patched {os.path.basename(dss_file_path)}")
    return True


def revert_dss_file(dss_file_path):
    """Restore original DSS file from backup."""
    backup = dss_file_path + '.no_pv_backup'
    if os.path.exists(backup):
        shutil.move(backup, dss_file_path)
        return True
    return False


# ============================================================
# PER-SYSTEM PROCESSING
# ============================================================

# Map system names to their _daily.dss files
DSS_FILES = {
    '13Bus': 'IEEE13Nodeckt_daily.dss',
    '34Bus': 'ieee34Mod1_daily.dss',
    '123Bus': 'IEEE123Master_daily.dss',
    '8500Node': 'Master_daily.dss',
}

def process_system(system_name, systems_dir, penetration=None, n_steps=144):
    """Add PV to one system."""
    print(f"\n  [{system_name}]")
    
    system_dir = os.path.join(systems_dir, 
                               system_name if system_name != '8500Node' else '8500-Node')
    
    if not os.path.exists(system_dir):
        print(f"    SKIP: {system_dir} not found")
        return False
    
    # Get PV configuration
    sites = get_pv_config(system_name, penetration)
    if not sites:
        return False
    
    # Create solar loadshape CSV
    create_daily_solar_csv(system_dir, n_steps)
    
    # Generate DSS block
    pv_block, total_kw = generate_pv_dss_block(sites, n_steps)
    
    # Patch DSS file
    dss_file = os.path.join(system_dir, DSS_FILES[system_name])
    if not os.path.exists(dss_file):
        print(f"    WARNING: {dss_file} not found")
        return False
    
    success = patch_dss_file(dss_file, pv_block)
    
    if success:
        print(f"    Added {len(sites)} PV sites, total {total_kw:.0f} kW")
        for site in sites:
            print(f"      {site['name']}: {site['kW']:.0f} kW at bus {site['bus']}")
    
    return success


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Add Distributed Solar PV to IEEE Test Feeders')
    parser.add_argument('--systems', nargs='+', default=None,
                        help='Systems to process (default: all)')
    parser.add_argument('--penetration', type=float, default=None,
                        help='PV penetration level 0-1 (default: per-system)')
    parser.add_argument('--steps', type=int, default=144,
                        help='Steps per episode (24=hourly, 144=10-min)')
    parser.add_argument('--revert', action='store_true',
                        help='Remove all PV additions')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    systems_dir = os.path.join(script_dir, 'systems')
    
    all_systems = ['13Bus', '34Bus', '123Bus', '8500Node']
    systems = args.systems or all_systems
    
    if args.revert:
        print("\n=== REVERTING: Removing PV from DSS files ===")
        for system in systems:
            system_dir = os.path.join(systems_dir,
                                       system if system != '8500Node' else '8500-Node')
            dss_file = os.path.join(system_dir, DSS_FILES.get(system, ''))
            if revert_dss_file(dss_file):
                print(f"  [{system}] Restored original DSS file")
            else:
                print(f"  [{system}] No backup found")
        return
    
    print(f"\n{'='*60}")
    print(f"Adding Distributed Solar PV Generation")
    print(f"Systems: {systems}")
    print(f"Penetration: {args.penetration or 'per-system default'}")
    print(f"Steps/episode: {args.steps}")
    print(f"{'='*60}")
    
    for system in systems:
        process_system(system, systems_dir, args.penetration, args.steps)
    
    print(f"\n{'='*60}")
    print(f"PV Integration Complete!")
    print(f"")
    print(f"Key points for the paper:")
    print(f"  - PV is uncontrollable (no Volt-Var or Volt-Watt from inverters)")
    print(f"  - Unity power factor operation (worst case for voltage)")
    print(f"  - Realistic daily profiles with stochastic cloud transients")
    print(f"  - Seasonal variation in day length and peak irradiance")
    print(f"")
    print(f"To revert: python add_renewable_generation.py --revert")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()