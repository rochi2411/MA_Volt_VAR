"""
prepare_step_profiles.py - Prepare 10-minute Resolution Load Profiles
==========================================================================
Converts the existing hourly LoadShape CSVs (8760 points = 365 days × 24h)
to 10-minute resolution (52560 points = 365 days × 144 steps) using cubic
spline interpolation.

This enables 144-step episodes covering a full 24-hour daily cycle at
10-minute control intervals — matching real utility SCADA/DMS refresh rates.

After running this script:
  1. LoadShape CSVs are upsampled from 8760 → 52560 points
  2. Old per-episode folders (000, 001, ...) are deleted for regeneration
  3. env_register.py is patched: max_episode_steps: 24 → 144
  4. loadprofile.py is patched: hardcoded stepsize=3600 → dynamic

Usage:
    python prepare_step_profiles.py                    # All systems
    python prepare_step_profiles.py --systems 13Bus    # One system
    python prepare_step_profiles.py --revert           # Undo changes
"""

import os
import sys
import shutil
import argparse
import numpy as np
from scipy import interpolate

sys.path.append('./powergym')

SYSTEMS = ['13Bus', '34Bus', '123Bus', '8500Node']
UPSAMPLE_FACTOR = 6  # 24 → 144 (hourly → 10-minute)
NEW_STEPS = 144

# Map env name to actual directory name under systems/
SYSTEM_DIRS = {
    '13Bus': '13Bus',
    '34Bus': '34Bus',
    '123Bus': '123Bus',
    '8500Node': '8500-Node',
}


def interpolate_loadshape(input_csv, output_csv, factor=6):
    """
    Upsample a LoadShape CSV from hourly to 10-minute resolution
    using cubic spline interpolation.
    
    8760 hourly points → 52560 ten-minute points
    
    Cubic spline is standard for load curve interpolation in power systems
    (smooth, preserves daily peaks/valleys, no negative values).
    """
    data = np.loadtxt(input_csv)
    n_original = len(data)
    n_new = n_original * factor
    
    # Original time indices (hourly)
    t_original = np.arange(n_original)
    
    # New time indices (10-minute)
    t_new = np.linspace(0, n_original - 1, n_new)
    
    # Cubic spline interpolation (not-a-knot: standard for non-periodic data)
    cs = interpolate.CubicSpline(t_original, data, bc_type='not-a-knot')
    data_interpolated = cs(t_new)
    
    # Ensure non-negative (load multipliers must be ≥ 0)
    data_interpolated = np.maximum(data_interpolated, 0.0)
    
    # Save
    np.savetxt(output_csv, data_interpolated, fmt='%.15f')
    
    return n_original, n_new


def prepare_system(system_name, systems_dir):
    """Prepare one system for 144-step episodes."""
    dir_name = SYSTEM_DIRS.get(system_name, system_name)
    loadshape_dir = os.path.join(systems_dir, dir_name, 'loadshape')
    
    if not os.path.exists(loadshape_dir):
        print(f"  SKIP: {loadshape_dir} not found")
        return False
    
    # --- Step 1: Backup and interpolate LoadShape CSVs ---
    # Match PowerGym's own discovery pattern: any CSV with 'loadshape' in the name
    csv_files = [f for f in os.listdir(loadshape_dir)
                 if 'loadshape' in f.lower() and f.lower().endswith('.csv')]
    
    # Exclude solar shape files and backup files
    csv_files = [f for f in csv_files
                 if 'solar' not in f.lower() and 'backup' not in f.lower()]
    
    if not csv_files:
        print(f"  SKIP: No LoadShape CSVs found in {loadshape_dir}")
        return False
    
    print(f"\n  [{system_name}] Processing {len(csv_files)} LoadShape files...")
    
    for csv_file in csv_files:
        csv_path = os.path.join(loadshape_dir, csv_file)
        backup_path = csv_path + '.hourly_backup'
        
        # Check if already processed
        if os.path.exists(backup_path):
            print(f"    {csv_file}: already processed (backup exists), skipping")
            continue
        
        # Backup original
        shutil.copy2(csv_path, backup_path)
        
        # Interpolate
        n_orig, n_new = interpolate_loadshape(csv_path, csv_path, UPSAMPLE_FACTOR)
        print(f"    {csv_file}: {n_orig} → {n_new} points (×{UPSAMPLE_FACTOR})")
    
    # --- Step 2: Delete old per-episode folders (will be regenerated) ---
    deleted_count = 0
    for item in os.listdir(loadshape_dir):
        item_path = os.path.join(loadshape_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            shutil.rmtree(item_path)
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"    Deleted {deleted_count} old episode folders (will regenerate)")
    
    # --- Step 3: Delete scale.txt to force regeneration ---
    scale_txt = os.path.join(loadshape_dir, 'scale.txt')
    if os.path.exists(scale_txt):
        os.remove(scale_txt)
        print(f"    Deleted scale.txt (forces profile regeneration)")
    
    return True


def revert_system(system_name, systems_dir):
    """Revert a system back to original hourly profiles."""
    dir_name = SYSTEM_DIRS.get(system_name, system_name)
    loadshape_dir = os.path.join(systems_dir, dir_name, 'loadshape')
    
    if not os.path.exists(loadshape_dir):
        return False
    
    restored = 0
    for f in os.listdir(loadshape_dir):
        if f.endswith('.hourly_backup'):
            original_name = f.replace('.hourly_backup', '')
            backup_path = os.path.join(loadshape_dir, f)
            original_path = os.path.join(loadshape_dir, original_name)
            shutil.move(backup_path, original_path)
            restored += 1
    
    # Delete episode folders for regeneration
    for item in os.listdir(loadshape_dir):
        item_path = os.path.join(loadshape_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            shutil.rmtree(item_path)
    
    scale_txt = os.path.join(loadshape_dir, 'scale.txt')
    if os.path.exists(scale_txt):
        os.remove(scale_txt)
    
    if restored > 0:
        print(f"  [{system_name}] Restored {restored} hourly backups")
    return restored > 0


def patch_env_register(env_register_path):
    """
    Patch env_register.py: change max_episode_steps from 24 to 144.
    """
    with open(env_register_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "'max_episode_steps': 144" in content:
        print("  env_register.py: already patched to 144 steps")
        return
    
    # Backup
    backup = env_register_path + '.24step_backup'
    if not os.path.exists(backup):
        shutil.copy2(env_register_path, backup)
    
    # Replace all occurrences
    new_content = content.replace("'max_episode_steps': 24", "'max_episode_steps': 144")
    
    count = content.count("'max_episode_steps': 24")
    with open(env_register_path, 'w') as f:
        f.write(new_content)
    
    print(f"  env_register.py: patched {count} entries (24 → 144 steps)")


def patch_loadprofile(loadprofile_path):
    """
    Patch loadprofile.py: fix hardcoded stepsize=3600 to be dynamic.
    
    Line 95: 'Set mode=Daily number=1 hour=0 stepsize=3600 sec=0'
    →        f'Set mode=Daily number=1 hour=0 stepsize={86400//self.steps} sec=0'
    """
    with open(loadprofile_path, 'r') as f:
        content = f.read()
    
    old_line = "fout.write('Set mode=Daily number=1 hour=0 stepsize=3600 sec=0\\n')"
    new_line = "fout.write(f'Set mode=Daily number=1 hour=0 stepsize={86400//self.steps} sec=0\\n')"
    
    if new_line in content:
        print("  loadprofile.py: already patched (dynamic stepsize)")
        return
    
    if old_line not in content:
        # Try alternate formatting
        old_line2 = 'fout.write(\'Set mode=Daily number=1 hour=0 stepsize=3600 sec=0\\n\')'
        if old_line2 in content:
            old_line = old_line2
        else:
            print("  loadprofile.py: WARNING — could not find stepsize line to patch")
            print("    Please manually change line 95:")
            print(f"    OLD: stepsize=3600")
            print(f"    NEW: stepsize={{86400//self.steps}}")
            return
    
    # Backup
    backup = loadprofile_path + '.original_backup'
    if not os.path.exists(backup):
        shutil.copy2(loadprofile_path, backup)
    
    new_content = content.replace(old_line, new_line)
    with open(loadprofile_path, 'w') as f:
        f.write(new_content)
    
    print("  loadprofile.py: patched stepsize=3600 → dynamic stepsize={86400//self.steps}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare 144-step (10-min resolution) load profiles')
    parser.add_argument('--systems', nargs='+', default=None,
                        help='Systems to process (default: all)')
    parser.add_argument('--revert', action='store_true',
                        help='Revert to original hourly profiles')
    parser.add_argument('--skip_patch', action='store_true',
                        help='Skip patching env_register.py and loadprofile.py')
    args = parser.parse_args()
    
    # Locate directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    systems_dir = os.path.join(script_dir, 'systems')
    powergym_dir = os.path.join(script_dir, 'powergym')
    
    systems = args.systems or SYSTEMS
    
    if args.revert:
        print("\n=== REVERTING to hourly profiles (24-step) ===")
        for system in systems:
            revert_system(system, systems_dir)
        print("\nDone. Remember to also revert env_register.py and loadprofile.py")
        print("  (restore from .24step_backup and .original_backup)")
        return
    
    print(f"\n{'='*60}")
    print(f"Preparing 144-step (10-min) Load Profiles")
    print(f"Systems: {systems}")
    print(f"Interpolation: Cubic spline (hourly → 10-min)")
    print(f"{'='*60}")
    
    # --- Process each system ---
    for system in systems:
        prepare_system(system, systems_dir)
    
    # --- Patch source files ---
    if not args.skip_patch:
        print(f"\n--- Patching configuration files ---")
        
        env_register = os.path.join(powergym_dir, 'env_register.py')
        if os.path.exists(env_register):
            patch_env_register(env_register)
        else:
            print(f"  WARNING: {env_register} not found")
        
        loadprofile = os.path.join(powergym_dir, 'loadprofile.py')
        if os.path.exists(loadprofile):
            patch_loadprofile(loadprofile)
        else:
            print(f"  WARNING: {loadprofile} not found")
    
    # --- Update training steps recommendation ---
    print(f"\n--- Recommended TIMESTEPS update for run_exp.py ---")
    print(f"With 144-step episodes (6× longer), increase training budgets:")
    print(f'  TIMESTEPS = {{')
    print(f'      "13Bus": 200000,     # was 50000')
    print(f'      "34Bus": 400000,     # was 100000')
    print(f'      "123Bus": 800000,    # was 200000')
    print(f'      "8500Node": 2000000, # was 500000')
    print(f'  }}')
    
    print(f"\n--- Summary ---")
    print(f"  ✓ LoadShape CSVs interpolated: 8760 → 52560 points")
    print(f"  ✓ Old episode folders deleted (auto-regenerate on first run)")
    print(f"  ✓ env_register.py: max_episode_steps = 144")
    print(f"  ✓ loadprofile.py: stepsize = 86400/144 = 600s (10 min)")
    print(f"  ! Update TIMESTEPS in run_exp.py (see above)")
    print(f"  ! Backups saved as .hourly_backup and .24step_backup")
    print(f"\nTo revert: python prepare_144step_profiles.py --revert")


if __name__ == "__main__":
    main()