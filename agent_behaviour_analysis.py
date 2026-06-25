"""
agent_behaviour_analysis.py - Agent Behaviour Analysis (v2)
================================================================
Compares all 7 methods: Specialist, Monolithic, MAPPO, MADDPG,
Heuristic, OpenDSS Auto, and Model-Based OPF.

Usage:
    python agent_behaviour_analysis.py --env_name 13Bus
    python agent_behaviour_analysis.py --env_name all
"""

import os, sys, json, numpy as np, matplotlib.pyplot as plt, argparse, torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)

sys.path.append('./powergym')
from stable_baselines3 import PPO
from powergym.env_register import make_env
from train_marl import IPPO_Wrapper
import gymnasium as gym

try:
    from heuristic_agent import HeuristicController, GymnasiumCompatibilityWrapper
except ImportError:
    HeuristicController = None

try:
    from train_mappo import MAPPO_Wrapper
except ImportError:
    MAPPO_Wrapper = None

try:
    from train_maddpg import MADDPGAgent, MADDPGEnvWrapper, Actor as MADDPGActor
except ImportError:
    MADDPGAgent = None

try:
    from opendss_auto_baseline import OpenDSSAutoController, GreedyVoltageOptimizer     
except ImportError:
    OpenDSSAutoController = None
    GreedyVoltageOptimizer = None

plt.style.use('seaborn-v0_8-whitegrid')
from paper_style import apply_paper_style
apply_paper_style()  # consistent sans-serif fonts + 300 DPI / TrueType export

EPISODE_STEPS = 144
HOURS_PER_STEP = 1.0 / 6
EPISODE_HOURS = 24.0

ALL_METHODS = ['specialist', 'monolithic', 'mappo', 'maddpg',
               'heuristic', 'opendss_auto', 'model_opf']

METHOD_NAMES = {
    'specialist': 'Specialist (Ours)', 'monolithic': 'Monolithic PPO',
    'mappo': 'MAPPO', 'maddpg': 'MADDPG', 'heuristic': 'Heuristic',
    'opendss_auto': 'OpenDSS Auto', 'model_opf': 'Model-Based OPF',
}
METHOD_COLORS = {
    'specialist': '#2ecc71', 'monolithic': '#3498db', 'mappo': '#9b59b6',
    'maddpg': '#f39c12', 'heuristic': '#e74c3c', 'opendss_auto': '#2c3e50',
    'model_opf': '#8c564b',
}
METHOD_LS = {
    'specialist': '-', 'monolithic': '--', 'mappo': '-.', 'maddpg': ':',
    'heuristic': '--', 'opendss_auto': '-.', 'model_opf': (0, (3, 1, 1, 1)),
}
METHOD_LW = {
    'specialist': 2.5, 'monolithic': 2.0, 'mappo': 2.0, 'maddpg': 2.0,
    'heuristic': 1.5, 'opendss_auto': 1.5, 'model_opf': 1.5,
}

DEVICE_CONFIG = {
    '13Bus': {
        'n_caps': 2, 'n_regs': 3, 'n_bats': 1,
        'cap_names': ['Cap 675 (3-Ph)', 'Cap 611 (1-Ph)'],
        'reg_names': ['Reg 1 (Ph A)', 'Reg 2 (Ph B)', 'Reg 3 (Ph C)'],
        'bat_names': ['BESS 680'],
    },
    '34Bus': {
        'n_caps': 2, 'n_regs': 6, 'n_bats': 2,
        'cap_names': ['Cap 844', 'Cap 848'],
        'reg_names': ['Reg1 Ph-A', 'Reg1 Ph-B', 'Reg1 Ph-C',
                      'Reg2 Ph-A', 'Reg2 Ph-B', 'Reg2 Ph-C'],
        'bat_names': ['BESS 890', 'BESS 832'],
    },
    '123Bus': {
        'n_caps': 4, 'n_regs': 7, 'n_bats': 4,
        'cap_names': ['C83 (3-Ph)', 'C88 (Ph-A)', 'C90 (Ph-B)', 'C92 (Ph-C)'],
        'reg_names': ['Reg1 (Gang)', 'Reg2 (Ph-A)', 'Reg3 (Ph-A)', 'Reg3 (Ph-C)',
                      'Reg4 (Ph-A)', 'Reg4 (Ph-B)', 'Reg4 (Ph-C)'],
        'bat_names': ['BESS 33', 'BESS 114', 'BESS 67', 'BESS 300'],
    },
    '8500Node': {
        'n_caps': 10, 'n_regs': 12, 'n_bats': 10,
        'cap_names': [f'Cap {i+1}' for i in range(10)],
        'reg_names': [f'Reg {i+1}' for i in range(12)],
        'bat_names': [f'BESS {i+1}' for i in range(10)],
    },
}

# ============================================================
# HELPERS
# ============================================================

def _init_data(config):
    return {
        'timesteps': [], 'hours': [], 'rewards': [], 'violations': [],
        'power_loss': [], 'voltage_min': [], 'voltage_max': [], 'voltage_mean': [],
        'cap_actions': {f'cap_{i}': [] for i in range(config['n_caps'])},
        'reg_actions': {f'reg_{i}': [] for i in range(config['n_regs'])},
        'bat_actions': {f'bat_{i}': [] for i in range(config['n_bats'])},
    }

def _parse_action(action, config):
    a = np.atleast_1d(action).flatten()
    idx = 0
    caps, regs, bats = [], [], []
    for _ in range(config['n_caps']):
        caps.append(int(a[idx]) % 2 if idx < len(a) else 0); idx += 1
    for _ in range(config['n_regs']):
        regs.append(int(a[idx]) % 33 if idx < len(a) else 16); idx += 1
    for _ in range(config['n_bats']):
        v = float(a[idx]) if idx < len(a) else 0.0
        if abs(v) > 1: v = (v / 16.0) - 1.0
        bats.append(v); idx += 1
    return caps, regs, bats

def _record(data, step, reward, info, obs, caps, regs, bats, config):
    data['timesteps'].append(step)
    data['hours'].append(step * HOURS_PER_STEP)
    data['rewards'].append(reward)
    vm = (obs >= 0.7) & (obs <= 1.3)
    vs = obs[vm] if np.any(vm) else np.array([1.0])
    data['voltage_min'].append(float(np.min(vs)))
    data['voltage_max'].append(float(np.max(vs)))
    data['voltage_mean'].append(float(np.mean(vs)))
    if isinstance(info, dict):
        data['violations'].append(info.get('constraint_cost', abs(info.get('vol_reward', 0.0))))
        data['power_loss'].append(info.get('power_loss_ratio', 0.0))
    else:
        data['violations'].append(0.0); data['power_loss'].append(0.0)
    for i in range(config['n_caps']):
        data['cap_actions'][f'cap_{i}'].append(caps[i] if i < len(caps) else 0)
    for i in range(config['n_regs']):
        data['reg_actions'][f'reg_{i}'].append(regs[i] if i < len(regs) else 16)
    for i in range(config['n_bats']):
        data['bat_actions'][f'bat_{i}'].append(bats[i] if i < len(bats) else 0.0)

# ============================================================
# COLLECTORS
# ============================================================

def _collect_ippo_style(env_name, model, wrapper_cls, config, seed):
    """Collect from IPPO-style sequential wrapper (Specialist or MAPPO)."""
    env = wrapper_cls(env_name)
    data = _init_data(config)
    obs, _ = env.reset(seed=seed)
    agents = env.agents
    n_agents = len(agents)
    at = env.ma_env.agent_types
    cc = rc = bc = 0
    a2i = {}
    for aid in agents:
        t = at[aid]
        if t == 'cap': a2i[aid] = ('cap', cc); cc += 1
        elif t == 'reg': a2i[aid] = ('reg', rc); rc += 1
        elif t == 'bat': a2i[aid] = ('bat', bc); bc += 1

    for step in range(EPISODE_STEPS):
        aa = {}
        fr, fi, done = 0, {}, False
        for ai in range(n_agents):
            action, _ = model.predict(obs, deterministic=True)
            aa[agents[ai]] = int(action)
            obs, r, te, tr, info = env.step(action)
            if ai == n_agents - 1: fr, fi, done = r, info, te or tr
        cd = [0]*config['n_caps']; rd = [16]*config['n_regs']; bd = [0.0]*config['n_bats']
        for aid, act in aa.items():
            dt, di = a2i[aid]
            if dt == 'cap': cd[di] = act % 2
            elif dt == 'reg': rd[di] = act % 33
            elif dt == 'bat': bd[di] = (act / 16.0) - 1.0
        # Record TRUE bus voltages, not the normalized per-agent obs: the
        # latter never lands in [0.7,1.3] so _record would fall back to a
        # constant 1.0 (a flat-line artifact). Mirror the other collectors.
        raw = env.ma_env.raw_env
        wrapped = raw.wrap_obs(raw.obs) if hasattr(raw, 'wrap_obs') else obs
        _record(data, step, fr, fi, wrapped, cd, rd, bd, config)
        if done: break
    env.close()
    return data

def collect_specialist(env_name, model, config, seed=42):
    return _collect_ippo_style(env_name, model, IPPO_Wrapper, config, seed)

def collect_mappo(env_name, model, config, seed=42):
    if MAPPO_Wrapper is None: return None
    return _collect_ippo_style(env_name, model, MAPPO_Wrapper, config, seed)

def collect_joint(env_name, model, agent_type, config, seed=42):
    """Monolithic or Heuristic."""
    raw = make_env(env_name)
    env = GymnasiumCompatibilityWrapper(raw)
    data = _init_data(config)
    obs, info = env.reset(seed=seed)
    ctrl = None
    if agent_type == 'heuristic' and HeuristicController:
        ctrl = HeuristicController(env_name); ctrl.reset()
    for step in range(EPISODE_STEPS):
        if agent_type == 'heuristic': action = ctrl.get_action(obs, info)
        else: action, _ = model.predict(obs, deterministic=True)
        caps, regs, bats = _parse_action(action, config)
        obs, reward, te, tr, info = env.step(action)
        _record(data, step, reward, info, obs, caps, regs, bats, config)
        if te or tr: break
    env.close()
    return data

def collect_maddpg(env_name, pt_path, config, seed=42):
    if MADDPGAgent is None: return None
    # Load the checkpoint first: building the env chdir's into the system
    # folder, which would break a relative pt_path.
    ckpt = torch.load(os.path.abspath(pt_path), map_location='cpu')
    # MADDPGEnvWrapper takes the env *name* and builds its own MultiAgentPowerGrid.
    env = MADDPGEnvWrapper(env_name)
    ma = env.ma_env
    data = _init_data(config)
    # Checkpoints store one actor state_dict per agent as {"actor_i": ...},
    # rebuilt exactly as train_maddpg.evaluate_maddpg does.
    actors = []
    for i, aid in enumerate(env.agents):
        actor = MADDPGActor(env.obs_dim, env.act_dims[aid])
        actor.load_state_dict(ckpt[f"actor_{i}"]); actor.eval()
        actors.append(actor)
    # Map each agent to its device-type slot, matching the other collectors.
    at = ma.agent_types
    a2i = {}; cc = rc = bc = 0
    for aid in env.agents:
        t = at[aid]
        if t == 'cap': a2i[aid] = ('cap', cc); cc += 1
        elif t == 'reg': a2i[aid] = ('reg', rc); rc += 1
        elif t == 'bat': a2i[aid] = ('bat', bc); bc += 1
    # Reset on the same load profile as every other method for a fair comparison.
    go = ma.raw_env.reset(load_profile_idx=seed % ma.raw_env.num_profiles)
    env.current_obs_dict = ma._extract_local_obs(go)
    obs_list = [env._get_local_obs(aid) for aid in env.agents]
    for step in range(EPISODE_STEPS):
        actions = []
        for i, actor in enumerate(actors):
            obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0)
            with torch.no_grad():
                act_oh = actor.get_action(obs_t, explore=False)
            actions.append(int(act_oh.argmax(-1).item()))
        obs_list, _global, reward, done, info = env.step(actions)
        caps = [0]*config['n_caps']; regs = [16]*config['n_regs']; bats = [0.0]*config['n_bats']
        for i, aid in enumerate(env.agents):
            dt, di = a2i[aid]; act = actions[i]
            if dt == 'cap': caps[di] = act % 2
            elif dt == 'reg': regs[di] = act % 33
            elif dt == 'bat': bats[di] = (act / 16.0) - 1.0
        # Record TRUE bus voltages. env._get_global_obs() returns NORMALIZED
        # voltages ((V-1)*10, centered at 0), which never land in _record's
        # [0.7,1.3] window and would collapse to a flat-1.0 artifact.
        wrapped = ma.raw_env.wrap_obs(ma.raw_env.obs) if hasattr(ma.raw_env, 'wrap_obs') else env._get_global_obs()
        _record(data, step, reward, info, wrapped, caps, regs, bats, config)
        if done: break
    return data

def collect_opendss_auto(env_name, config, seed=42):
    if OpenDSSAutoController is None: return None
    raw = make_env(env_name, dss_act=True)
    ctrl = OpenDSSAutoController(raw)
    data = _init_data(config)
    raw.reset(load_profile_idx=seed % raw.num_profiles)
    for step in range(EPISODE_STEPS):
        ba = ctrl.get_battery_actions()
        if raw.bat_num > 0:
            for i, bn in enumerate(raw.bat_names):
                raw.circuit.batteries[bn].step_before_solve(ba[i])
        obs, reward, done, info = raw.dss_step()
        if raw.bat_num > 0:
            for bn in raw.bat_names: raw.circuit.batteries[bn].step_after_solve()
        caps = [int(raw.circuit.capacitors[c].status) for c in raw.cap_names] if hasattr(raw, 'cap_names') else []
        regs = []
        if hasattr(raw, 'reg_names'):
            for r in raw.reg_names:
                reg = raw.circuit.regulators[r]
                mintap, maxtap, numtaps = reg.tap_feature
                step = (maxtap - mintap) / numtaps if numtaps > 0 else 1
                tap_idx = int(round((reg.tap - mintap) / step))
                regs.append(max(0, min(32, tap_idx)))
        else:
            regs = [16] * config['n_regs']
        bats_n = [(b / 16.0) - 1.0 if isinstance(b, (int, np.integer)) else float(b) for b in ba]
        wrapped = raw.wrap_obs(raw.obs) if hasattr(raw, 'wrap_obs') else np.array([1.0])
        _record(data, step, reward, info, wrapped, caps, regs, bats_n, config)
        if done: break
    return data

def collect_model_opf(env_name, config, seed=42):
    """Roll out the LinDist3Flow MILP oracle (the actual 'Model-based OPF'
    baseline) and record its trajectory. Solves the perfect-information
    144-period program, then replays the resulting device schedule in the env.
    """
    try:
        from opf_lindist3flow import (NetworkModel, OPFBuilder, load_horizon_profiles,
                                      slack_voltage_sq, get_weights, _project_battery_state)
    except ImportError:
        return None
    raw = make_env(env_name, dss_act=False)
    nm = NetworkModel(raw)
    idx = seed % raw.num_profiles
    p_dem, q_dem = load_horizon_profiles(nm, raw, idx, EPISODE_STEPS)
    raw.reset(load_profile_idx=idx)
    builder = OPFBuilder(nm, p_dem, q_dem, EPISODE_STEPS, get_weights(env_name),
                         slack_voltage_sq(nm), tap_tier=1)
    builder.solve(time_limit=300, mip_gap=0.01)
    traj = builder.extract_trajectory()
    cap_n, reg_n, bat_n = len(nm.caps), len(nm.regs), len(nm.bats)
    data = _init_data(config)
    raw.reset(load_profile_idx=idx)
    for step in range(EPISODE_STEPS):
        action = [int(traj['caps'][k, step]) for k in range(cap_n)]
        action += [int(traj['regs'][k, step]) for k in range(reg_n)]
        for k in range(bat_n):
            action.append(_project_battery_state(nm.bats[k], traj['bats'][k, step]))
        has_float = any(isinstance(a, float) for a in action)
        obs, reward, done, info = raw.step(
            np.array(action, dtype=float) if has_float else np.array(action, dtype=int))
        # Record devices directly from the (normalized) OPF schedule.
        caps = [int(traj['caps'][k, step]) for k in range(cap_n)]
        regs = [int(traj['regs'][k, step]) for k in range(reg_n)]
        bats = [float(np.clip(traj['bats'][k, step] / max(nm.bats[k]['max_kw'], 1e-9),
                              -1.0, 1.0)) for k in range(bat_n)]
        wrapped = raw.wrap_obs(raw.obs) if hasattr(raw, 'wrap_obs') else obs
        _record(data, step, reward, info, wrapped, caps, regs, bats, config)
        if done: break
    return data

# ============================================================
# ORCHESTRATOR
# ============================================================

def collect_all(env_name, results_dir, seed=42):
    config = DEVICE_CONFIG[env_name]
    data = {}

    # Resolve to absolute path
    results_dir = os.path.abspath(results_dir)
    print(f"  Results dir: {results_dir}")

    def me(p):
        """Check if model exists, print status."""
        exists = os.path.exists(p + '.zip') or os.path.exists(p)
        status = "FOUND" if exists else "NOT FOUND"
        print(f"    {os.path.basename(p)}: {status}")
        return exists

    # 1. Specialist
    sp = f"{results_dir}/specialist_{env_name}_seed{seed}"
    if me(sp):
        try:
            data['specialist'] = collect_specialist(env_name, PPO.load(sp), config, seed)
            print(f"      -> {len(data['specialist']['timesteps'])} steps collected")
        except Exception as e:
            print(f"      -> ERROR: {e}")

    # 2. Monolithic
    mo = f"{results_dir}/monolithic_{env_name}_seed{seed}"
    if me(mo):
        try:
            data['monolithic'] = collect_joint(env_name, PPO.load(mo), 'monolithic', config, seed)
            print(f"      -> {len(data['monolithic']['timesteps'])} steps collected")
        except Exception as e:
            print(f"      -> ERROR: {e}")

    # 3. MAPPO
    ma = f"{results_dir}/mappo_{env_name}_seed{seed}"
    if me(ma):
        if MAPPO_Wrapper:
            try:
                data['mappo'] = collect_mappo(env_name, PPO.load(ma), config, seed)
                print(f"      -> {len(data['mappo']['timesteps'])} steps collected")
            except Exception as e:
                print(f"      -> ERROR: {e}")
        else:
            print(f"      -> SKIPPED (MAPPO_Wrapper not imported)")

    # 4. MADDPG
    md = f"{results_dir}/maddpg_{env_name}_seed{seed}.pt"
    md_exists = os.path.exists(md)
    print(f"    maddpg_{env_name}_seed{seed}.pt: {'FOUND' if md_exists else 'NOT FOUND'}")
    if md_exists:
        if MADDPGAgent:
            try:
                data['maddpg'] = collect_maddpg(env_name, md, config, seed)
                print(f"      -> {len(data['maddpg']['timesteps'])} steps collected")
            except Exception as e:
                print(f"      -> ERROR: {e}")
        else:
            print(f"      -> SKIPPED (MADDPGAgent not imported)")

    # 5. Heuristic
    if HeuristicController:
        print(f"    Heuristic: RUNNING")
        try:
            data['heuristic'] = collect_joint(env_name, None, 'heuristic', config, seed)
            print(f"      -> {len(data['heuristic']['timesteps'])} steps collected")
        except Exception as e:
            print(f"      -> ERROR: {e}")
    else:
        print(f"    Heuristic: SKIPPED (not imported)")

    # 6. OpenDSS Auto
    if OpenDSSAutoController:
        print(f"    OpenDSS Auto: RUNNING")
        try:
            r = collect_opendss_auto(env_name, config, seed)
            if r:
                data['opendss_auto'] = r
                print(f"      -> {len(data['opendss_auto']['timesteps'])} steps collected")
        except Exception as e:
            print(f"      -> ERROR: {e}")
    else:
        print(f"    OpenDSS Auto: SKIPPED (not imported)")

    # 7. Model-Based OPF
    if GreedyVoltageOptimizer:
        print(f"    Model-Based OPF: RUNNING")
        try:
            r = collect_model_opf(env_name, config, seed)
            if r:
                data['model_opf'] = r
                print(f"      -> {len(data['model_opf']['timesteps'])} steps collected")
        except Exception as e:
            print(f"      -> ERROR: {e}")
    else:
        print(f"    Model-Based OPF: SKIPPED (not imported)")

    # Summary
    print(f"\n  Methods collected: {list(data.keys())}")
    missing = [m for m in ALL_METHODS if m not in data]
    if missing:
        print(f"  Missing: {missing}")

    for m in data:
        if data[m] is None: continue
        for k in ['timesteps','hours','rewards','violations','power_loss','voltage_min','voltage_max','voltage_mean']:
            data[m][k] = np.array(data[m][k])
    return data

# ============================================================
# PLOTS
# ============================================================

def _sty(m):
    return METHOD_COLORS.get(m,'#333'), METHOD_LS.get(m,'-'), METHOD_LW.get(m,1.5), METHOD_NAMES.get(m,m)

def _legend_outside(ax, ncol=1):
    """Place the legend just outside the axes (upper-right) so it never
    overlaps the data. _save_fig uses bbox_inches='tight', which expands the
    saved canvas to include the legend."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              ncol=ncol, frameon=True, framealpha=0.9, borderaxespad=0.0)

def _save_fig(path):
    """Ensure directory exists, then save and close."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def plot_regulators(dd, env, path):
    cfg = DEVICE_CONFIG[env]; nr = min(3, cfg['n_regs'])
    fig, axes = plt.subplots(nr, 1, figsize=(14, 3*nr), sharex=True)
    if nr == 1: axes = [axes]
    for i, ax in enumerate(axes):
        rk = f'reg_{i}'; rn = cfg['reg_names'][i] if i < len(cfg['reg_names']) else f'Reg {i+1}'
        for m, d in dd.items():
            if d is None or rk not in d['reg_actions']: continue
            c, ls, lw, lb = _sty(m)
            ax.step(d['hours'], d['reg_actions'][rk], color=c, linestyle=ls, linewidth=lw, label=lb, alpha=0.85, where='post')
        ax.set_ylabel(f'{rn}\nTap'); ax.set_ylim(0, 32); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (Hours)')
    for ax in axes: ax.set_xlim(0, EPISODE_HOURS)
    _legend_outside(axes[0])
    fig.suptitle(f'IEEE {env}: Regulator Control (24h)', y=1.02)
    plt.tight_layout(); _save_fig(path)

def plot_capacitors(dd, env, path):
    cfg = DEVICE_CONFIG[env]; nc = cfg['n_caps']
    fig, axes = plt.subplots(nc, 1, figsize=(14, 2.5*nc), sharex=True)
    if nc == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ck = f'cap_{i}'; cn = cfg['cap_names'][i] if i < len(cfg['cap_names']) else f'Cap {i+1}'
        for m, d in dd.items():
            if d is None or ck not in d['cap_actions']: continue
            c, ls, lw, lb = _sty(m)
            ax.step(d['hours'], d['cap_actions'][ck], color=c, linestyle=ls, linewidth=lw, label=lb, alpha=0.85, where='post')
        ax.set_ylabel(f'{cn}'); ax.set_ylim(-0.1, 1.1); ax.set_yticks([0,1]); ax.set_yticklabels(['OFF','ON'])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (Hours)')
    for ax in axes: ax.set_xlim(0, EPISODE_HOURS)
    _legend_outside(axes[0])
    fig.suptitle(f'IEEE {env}: Capacitor Switching (24h)', y=1.02)
    plt.tight_layout(); _save_fig(path)

def plot_batteries(dd, env, path):
    cfg = DEVICE_CONFIG[env]; nb = cfg['n_bats']
    fig, axes = plt.subplots(nb, 1, figsize=(14, 3*nb), sharex=True)
    if nb == 1: axes = [axes]
    for i, ax in enumerate(axes):
        bk = f'bat_{i}'; bn = cfg['bat_names'][i] if i < len(cfg['bat_names']) else f'BESS {i+1}'
        for m, d in dd.items():
            if d is None or bk not in d['bat_actions']: continue
            c, ls, lw, lb = _sty(m)
            ax.plot(d['hours'], d['bat_actions'][bk], color=c, linestyle=ls, linewidth=lw, label=lb, alpha=0.8)
        ax.axhline(0, color='black', lw=0.5); ax.set_ylabel(f'{bn}\nPower'); ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (Hours)')
    for ax in axes: ax.set_xlim(0, EPISODE_HOURS)
    _legend_outside(axes[0])
    fig.suptitle(f'IEEE {env}: Battery Dispatch (24h)', y=1.02)
    plt.tight_layout(); _save_fig(path)

def plot_voltage(dd, env, path):
    fig, ax = plt.subplots(figsize=(14, 5))
    for m, d in dd.items():
        if d is None: continue
        c, ls, lw, lb = _sty(m)
        ax.plot(d['hours'], d['voltage_mean'], color=c, linestyle=ls, linewidth=lw, label=lb, alpha=0.85)
    ax.axhline(1.05, color='red', ls='--', lw=1, alpha=0.6, label='_nolegend_')
    ax.axhline(0.95, color='red', ls='--', lw=1, alpha=0.6, label='_nolegend_')
    ax.fill_between([0, EPISODE_HOURS], 0.95, 1.05, alpha=0.08, color='green')
    ax.set_xlabel('Time (Hours)'); ax.set_ylabel('Mean Voltage (p.u.)')
    ax.set_ylim(0.90, 1.10); ax.set_xlim(0, EPISODE_HOURS)
    _legend_outside(ax); ax.grid(True, alpha=0.3)
    ax.set_title(f'IEEE {env}: 24h Voltage Profile')
    plt.tight_layout(); _save_fig(path)

def plot_cumulative(dd, env, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for m, d in dd.items():
        if d is None: continue
        c, ls, lw, lb = _sty(m)
        cr = np.cumsum(d['rewards'])
        ax.plot(d['hours'], cr, color=c, linestyle=ls, linewidth=lw, label=f"{lb} ({cr[-1]:.1f})", alpha=0.85)
    ax.set_xlabel('Time (Hours)'); ax.set_ylabel('Cumulative Reward')
    ax.set_title(f'IEEE {env}: Cumulative Reward (24h)'); ax.set_xlim(0, EPISODE_HOURS)
    _legend_outside(ax); ax.grid(True, alpha=0.3)
    plt.tight_layout(); _save_fig(path)

def plot_violations(dd, env, path):
    avail = [m for m in ALL_METHODS if m in dd and dd[m] is not None]
    n = len(avail)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(2.5*n, 3.5))
    if n == 1: axes = [axes]
    for ax, m in zip(axes, avail):
        d = dd[m]; bins = np.zeros((1, 24))
        for h, v in zip(d['hours'], d['violations']):
            hi = min(int(h), 23); bins[0, hi] = max(bins[0, hi], v)
        im = ax.imshow(bins, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=max(0.1, np.max(bins)), extent=[0,24,0,1])
        ax.set_xlabel('Hour'); ax.set_title(f'{METHOD_NAMES.get(m,m)}\n({np.sum(d["violations"]):.3f})', fontsize=9); ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Viol.')
    fig.suptitle(f'IEEE {env}: Violation Heatmap (24h)', fontsize=13, y=1.05)
    plt.tight_layout(); _save_fig(path)

# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

def _run_noisy_episode(env_name, model, agent_type, wrapper_cls, noise_pct, config, seed):
    """Run one episode with observation noise. Returns (total_reward, mean_violation)."""
    if agent_type in ('specialist', 'mappo'):
        env = wrapper_cls(env_name)
        obs, _ = env.reset(seed=seed)
        agents = env.agents
        n_agents = len(agents)
        ep_reward = 0
        violations = []
        for step in range(EPISODE_STEPS):
            fr, fi, done = 0, {}, False
            for ai in range(n_agents):
                noisy = obs + (noise_pct / 100.0) * np.random.randn(*obs.shape) if noise_pct > 0 else obs
                action, _ = model.predict(noisy, deterministic=True)
                obs, r, te, tr, info = env.step(action)
                if ai == n_agents - 1:
                    fr, fi, done = r, info, te or tr
            ep_reward += fr
            if isinstance(fi, dict):
                violations.append(fi.get('constraint_cost', abs(fi.get('vol_reward', 0.0))))
            if done: break
        env.close()
    elif agent_type == 'heuristic':
        raw = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw)
        obs, info = env.reset(seed=seed)
        ctrl = HeuristicController(env_name); ctrl.reset()
        ep_reward = 0; violations = []
        for step in range(EPISODE_STEPS):
            noisy = obs + (noise_pct / 100.0) * np.random.randn(*obs.shape) if noise_pct > 0 else obs
            action = ctrl.get_action(noisy, info)
            obs, reward, te, tr, info = env.step(action)
            ep_reward += reward
            if isinstance(info, dict):
                violations.append(info.get('constraint_cost', abs(info.get('vol_reward', 0.0))))
            if te or tr: break
        env.close()
    elif agent_type == 'monolithic':
        raw = make_env(env_name)
        env = GymnasiumCompatibilityWrapper(raw)
        obs, info = env.reset(seed=seed)
        ep_reward = 0; violations = []
        for step in range(EPISODE_STEPS):
            noisy = obs + (noise_pct / 100.0) * np.random.randn(*obs.shape) if noise_pct > 0 else obs
            action, _ = model.predict(noisy, deterministic=True)
            obs, reward, te, tr, info = env.step(action)
            ep_reward += reward
            if isinstance(info, dict):
                violations.append(info.get('constraint_cost', abs(info.get('vol_reward', 0.0))))
            if te or tr: break
        env.close()
    else:
        return 0.0, 0.0

    return ep_reward, np.mean(violations) if violations else 0.0


def run_robustness_analysis(env_name, results_dir, seed=42,
                            noise_levels=None, n_episodes=5):
    """
    Test all available methods under increasing observation noise.
    """
    if noise_levels is None:
        noise_levels = [0, 5, 10, 15, 20, 25]

    config = DEVICE_CONFIG[env_name]
    results = {}

    # Resolve to absolute path
    results_dir = os.path.abspath(results_dir)

    def me(p):
        exists = os.path.exists(p + '.zip') or os.path.exists(p)
        print(f"    {os.path.basename(p)}: {'FOUND' if exists else 'NOT FOUND'}")
        return exists

    # Build method -> (model, agent_type, wrapper_cls) mapping
    methods_to_test = {}

    sp = f"{results_dir}/specialist_{env_name}_seed{seed}"
    if me(sp):
        methods_to_test['specialist'] = (PPO.load(sp), 'specialist', IPPO_Wrapper)

    mo = f"{results_dir}/monolithic_{env_name}_seed{seed}"
    if me(mo):
        methods_to_test['monolithic'] = (PPO.load(mo), 'monolithic', None)

    ma = f"{results_dir}/mappo_{env_name}_seed{seed}"
    if me(ma):
        if MAPPO_Wrapper:
            methods_to_test['mappo'] = (PPO.load(ma), 'mappo', MAPPO_Wrapper)
        else:
            print(f"      -> SKIPPED (MAPPO_Wrapper not imported)")

    if HeuristicController:
        methods_to_test['heuristic'] = (None, 'heuristic', None)

    # Note: MADDPG, OpenDSS Auto, Model-Based OPF are excluded from
    # robustness analysis because they don't take raw observations as
    # input in the same way (MADDPG uses its own obs pipeline, model-based
    # methods read directly from the simulator)

    print(f"\n  Robustness analysis for {env_name}")
    print(f"  Methods: {list(methods_to_test.keys())}")
    print(f"  Noise levels: {noise_levels}%")
    print(f"  Episodes per level: {n_episodes}")

    for method, (model, atype, wcls) in methods_to_test.items():
        results[method] = {
            'reward_mean': [], 'reward_std': [],
            'violation_mean': [], 'violation_std': []
        }

        for noise in noise_levels:
            rewards, viols = [], []
            for ep in range(n_episodes):
                r, v = _run_noisy_episode(
                    env_name, model, atype, wcls, noise, config, seed=seed + ep)
                rewards.append(r)
                viols.append(v)

            results[method]['reward_mean'].append(float(np.mean(rewards)))
            results[method]['reward_std'].append(float(np.std(rewards)))
            results[method]['violation_mean'].append(float(np.mean(viols)))
            results[method]['violation_std'].append(float(np.std(viols)))

            print(f"    {method} @ {noise}% noise: reward={np.mean(rewards):.2f}±{np.std(rewards):.2f}")

    return {'noise_levels': noise_levels, 'methods': results}


def plot_robustness(rob_results, env_name, save_path):
    """Plot reward degradation and violation increase under noise."""
    noise_levels = rob_results['noise_levels']
    methods = rob_results['methods']

    if not methods:
        print(f"  No robustness data for {env_name}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward degradation
    ax1 = axes[0]
    for method, data in methods.items():
        c, ls, lw, lb = _sty(method)
        rm = np.array(data['reward_mean'])
        rs = np.array(data['reward_std'])
        # Normalize to % of baseline (0% noise)
        if rm[0] != 0:
            normalized = (rm / rm[0]) * 100
        else:
            normalized = rm
        ax1.plot(noise_levels, normalized, color=c, linestyle=ls, linewidth=lw,
                 marker='o', markersize=6, label=lb)
        ax1.fill_between(noise_levels,
                         ((rm - rs) / abs(rm[0]) * 100) if rm[0] != 0 else rm - rs,
                         ((rm + rs) / abs(rm[0]) * 100) if rm[0] != 0 else rm + rs,
                         color=c, alpha=0.15)

    ax1.set_xlabel('Observation Noise (%)')
    ax1.set_ylabel('Reward (% of baseline)')
    ax1.set_title('Reward Degradation Under Noise')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(100, color='gray', ls=':', lw=0.8)

    # Violation increase
    ax2 = axes[1]
    for method, data in methods.items():
        c, ls, lw, lb = _sty(method)
        vm = np.array(data['violation_mean'])
        vs = np.array(data['violation_std'])
        ax2.plot(noise_levels, vm, color=c, linestyle=ls, linewidth=lw,
                 marker='s', markersize=6, label=lb)
        ax2.fill_between(noise_levels, vm - vs, vm + vs, color=c, alpha=0.15)

    ax2.set_xlabel('Observation Noise (%)')
    ax2.set_ylabel('Mean Voltage Violation')
    ax2.set_title('Violation Increase Under Noise')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'IEEE {env_name}: Robustness to Observation Noise', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(save_path)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Agent Behavior Analysis (v2)')
    parser.add_argument('--env_name', type=str, default='13Bus', choices=['13Bus','34Bus','123Bus','all'])
    parser.add_argument('--results_dir', type=str, default='./experiment_results')
    parser.add_argument('--output_dir', type=str, default='./analysis_plots')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--robustness', action='store_true', help='Run robustness analysis under observation noise')
    parser.add_argument('--noise_levels', type=str, default='0,5,10,15,20,25',
                        help='Comma-separated noise percentages')
    parser.add_argument('--rob_episodes', type=int, default=5,
                        help='Episodes per noise level for robustness')
    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    envs = ['13Bus','34Bus','123Bus'] if args.env_name == 'all' else [args.env_name]
    noise_levels = [int(x) for x in args.noise_levels.split(',')]

    print(f"\n{'='*60}")
    print(f"Agent Behavior Analysis (v2) - 7 Methods, 24h Episodes")
    print(f"{'='*60}")

    for env in envs:
        print(f"\n# {env}")
        dd = collect_all(env, args.results_dir, args.seed)
        if not dd: print(f"  No data"); continue
        od = args.output_dir
        os.makedirs(od, exist_ok=True)
        plot_regulators(dd, env, f"{od}/regulator_{env}.png")
        plot_capacitors(dd, env, f"{od}/capacitor_{env}.png")
        plot_batteries(dd, env, f"{od}/battery_{env}.png")
        plot_voltage(dd, env, f"{od}/voltage_profile_{env}.png")
        plot_cumulative(dd, env, f"{od}/cumulative_reward_{env}.png")
        plot_violations(dd, env, f"{od}/violation_heatmap_{env}.png")
        with open(f"{od}/behavior_data_{env}.json", 'w') as f:
            json.dump(dd, f, indent=2, cls=NumpyEncoder)

        # Robustness analysis
        if args.robustness:
            rob = run_robustness_analysis(
                env, args.results_dir, args.seed,
                noise_levels=noise_levels, n_episodes=args.rob_episodes)
            plot_robustness(rob, env, f"{od}/robustness_{env}.png")
            with open(f"{od}/robustness_data_{env}.json", 'w') as f:
                json.dump(rob, f, indent=2, cls=NumpyEncoder)

    print(f"\nDone! Plots in {args.output_dir}")

if __name__ == "__main__":
    main()