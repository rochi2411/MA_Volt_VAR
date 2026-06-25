# Experiment Reproduction Guide

This guide provides detailed instructions for reproducing the experiments from the paper.

## Quick Start

### Single System, Single Seed
```bash
# Train Specialist Ensemble on IEEE 13-Bus
python train_marl.py --env_name 13Bus --steps 50000 --seed 42

# Evaluate and visualize
python agent_behaviour_analysis.py --env_name 13Bus
```

### Complete Experiments (All Methods, All Systems, Multiple Seeds)
```bash
# Run all experiments (5 seeds × 3 systems × 7 methods)
python run_exp.py --env all --seeds 5

# This will take several hours. Results saved to experiment_results/
```

## Detailed Experiment Workflow

### 1. Training Individual Methods

#### Specialist Ensemble (IPPO) - Our Method
```bash
python train_marl.py --env_name 13Bus --steps 50000 --seed 42
python train_marl.py --env_name 34Bus --steps 100000 --seed 42
python train_marl.py --env_name 123Bus --steps 150000 --seed 42
```

**Output**: Trained models saved flat into `experiment_results/marl_<env>_seed<seed>.zip`

#### Monolithic PPO Baseline
```bash
python train_monolithic.py --env_name 13Bus --steps 50000 --seed 42
python train_monolithic.py --env_name 34Bus --steps 100000 --seed 42
python train_monolithic.py --env_name 123Bus --steps 150000 --seed 42
```

#### MAPPO Baseline
```bash
python train_mappo.py --env_name 13Bus --steps 50000 --seed 42
python train_mappo.py --env_name 34Bus --steps 100000 --seed 42
python train_mappo.py --env_name 123Bus --steps 150000 --seed 42
```

#### MADDPG Baseline
```bash
python train_maddpg.py --env_name 13Bus --steps 50000 --seed 42
python train_maddpg.py --env_name 34Bus --steps 100000 --seed 42
python train_maddpg.py --env_name 123Bus --steps 150000 --seed 42
```

### 2. Evaluating Baselines

Heuristic, OpenDSS Auto, and OPF baselines don't require training:

```bash
# Evaluation happens automatically during run_exp.py
# Or manually evaluate:
python agent_behaviour_analysis.py --env_name 13Bus --methods heuristic,opendss,opf
```

### 3. Multi-Seed Experiments

For statistical significance:

```bash
# 5 seeds (paper results)
python run_exp.py --env 13Bus --seeds 5

# 10 seeds (extended validation)
python run_exp.py --env 13Bus --seeds 10

# Custom seed list
python run_exp.py --env 13Bus --seed_list 42,123,456,789,1011
```

### 4. Analysis and Visualization

#### Behavioral Analysis
```bash
# Generate all plots for one system
python agent_behaviour_analysis.py --env_name 13Bus

# Generate plots for all systems
python agent_behaviour_analysis.py --env_name all

# Include robustness analysis
python agent_behaviour_analysis.py --env_name 13Bus --robustness
```

**Generated Plots**:
- `analysis_plots/regulator_comparison_13Bus.png`
- `analysis_plots/capacitor_comparison_13Bus.png`
- `analysis_plots/battery_comparison_13Bus.png`
- `analysis_plots/voltage_profile_13Bus.png`
- `analysis_plots/cumulative_rewards_13Bus.png`
- `analysis_plots/violation_heatmap_13Bus.png`
- `analysis_plots/training_curve_13Bus.png`
- `analysis_plots/robustness_13Bus.png` (if --robustness flag used)

#### Statistical Analysis

Statistical analysis is automatically performed by `run_exp.py` using the `enhanced_statistics` module. Results include:
- Bootstrap confidence intervals (BCa method)
- Interquartile Mean (IQM)
- Mann-Whitney U tests
- Effect sizes (Cohen's d)
- Performance profiles

To view statistics, check the console output after running:
```bash
python run_exp.py --env all --seeds 5
```

#### Aggregate Results
```bash
# Aggregate multi-seed results from all environments
python aggregate_results.py --results_dir ./experiment_results

# Aggregate specific environments only
python aggregate_results.py --results_dir ./experiment_results --envs 13Bus 34Bus

# Specify custom output directory for plots
python aggregate_results.py --results_dir ./experiment_results --output_dir ./figures
```

**Note**: 
- All plotting functions are integrated into `run_exp.py` and `agent_behaviour_analysis.py`
- `enhanced_statistics.py` is a library module imported by other scripts, not a standalone command
- Results are automatically saved as CSV and displayed in console tables

### 5. Ablation Studies

```bash
# Run ablation experiments
python run_ablation.py --env_name 13Bus

# Ablation options:
# - No parameter sharing
# - No agent-type embedding
# - Different network architectures
# - Different hyperparameters
```

## Expected Results

The authoritative 5-seed performance numbers (reward and violation index for every method on
all three feeders) live in the [Performance Comparison](../README.md#performance-comparison)
section of the main README and in the paper. They are not duplicated here to avoid drift.

As a sanity check, a correct run should reproduce: Specialist reward of roughly **−3** (13-Bus),
**−15** (34-Bus), and **−3.6** (123-Bus), a > 80% reward improvement over Monolithic PPO, and the
Specialist coming within ~1 reward unit of the LinDist3Flow OPF oracle on the 34-Bus.

### Approximate Training Time (single seed, CPU)

| System | Steps | Specialist | Monolithic / MAPPO / MADDPG |
|--------|-------|-----------|------------------------------|
| 13-Bus | 50k   | ~15 min | ~20–30 min |
| 34-Bus | 100k  | ~30 min | ~45–60 min |
| 123-Bus | 150k | ~60 min | ~90–120 min |

*Indicative wall-clock on a modern multi-core CPU; varies with hardware. No GPU is required.*

## Hyperparameters

### Training Configuration

```python
# Specialist Ensemble (IPPO)
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [256, 256],
        "activation_fn": "tanh"
    }
}

# Monolithic PPO (same as above)

# MAPPO
{
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "centralized_critic": True,
    "critic_net_arch": [512, 512]
}

# MADDPG
{
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "buffer_size": 100000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "gumbel_temperature": 1.0
}
```

### Environment Configuration

```python
# Episode configuration
{
    "episode_length": 144,  # 144 timesteps (24-hour horizon at 10-min resolution)
    "voltage_limits": [0.95, 1.05],  # ANSI C84.1 Range A
    "reward_weights": {
        "voltage_violation": -100.0,
        "switching_penalty": -0.1,
        "battery_degradation": -0.05
    }
}
```

## Troubleshooting

**Important**: Hyperparameters are currently hardcoded in the training scripts. To modify them, you must edit the source files directly.

### Available Command-Line Arguments

All training scripts support these arguments:
```bash
python train_marl.py --env_name 13Bus --steps 50000 --seed 42 --verbose 1
```

- `--env_name`: Environment selection (`13Bus`, `34Bus`, `123Bus`)
- `--steps`: Total training timesteps (default: 50000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--model_name`: Custom model name (optional)
- `--verbose`: Verbosity level (0=silent, 1=progress)
- `--no_save_data`: Disable training data saving (flag)

### Training Divergence

If training diverges (rewards become very negative), modify hyperparameters in the source code:

**Edit `train_marl.py` (lines ~289-310)**:
```python
model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=1e-4,      # Reduce from 3e-4
    n_steps=2048,
    batch_size=128,          # Increase from 64
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,          # Reduce from 0.2
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
    seed=seed,
    verbose=verbose
)
```

Then run:
```bash
python train_marl.py --env_name 13Bus --steps 50000 --seed 42
```

### Out of Memory

For large systems (123-Bus), reduce memory usage:

**Edit `train_marl.py` (lines ~289-310)**:
```python
model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=3e-4,
    n_steps=1024,            # Reduce from 2048
    batch_size=32,           # Reduce from 64
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh),  # Smaller network
    seed=seed,
    verbose=verbose
)
```

### Slow Training

To speed up training:

**Edit `train_marl.py` (lines ~289-310)**:
```python
model = PPO(
    "MlpPolicy",
    env,
    device=device,
    learning_rate=3e-4,
    n_steps=1024,            # Reduce from 2048
    batch_size=64,
    n_epochs=5,              # Reduce from 10
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh),
    seed=seed,
    verbose=verbose
)
```

**Alternative**: Reduce total training steps:
```bash
python train_marl.py --env_name 13Bus --steps 25000 --seed 42
```

## Reproducing Paper Figures

### Figure 1: Architecture Diagram
Generated manually (not code-based)

### Figure 2: Training Curves
```bash
python agent_behaviour_analysis.py --env_name all --plot training_curves
```

### Figure 3: Voltage Profiles
```bash
python agent_behaviour_analysis.py --env_name all --plot voltage_profiles
```

### Figure 4: Control Actions
```bash
python agent_behaviour_analysis.py --env_name all --plot control_actions
```

### Figure 5: Statistical Comparison
```bash
python enhanced_statistics.py --env_name all --plot performance_profiles
```

### Figure 6: Robustness Analysis
```bash
python agent_behaviour_analysis.py --env_name all --robustness
```

### Table 1: Performance Summary
```bash
python aggregate_results.py --env_name all --format latex
```

### Table 2: Statistical Tests
```bash
python enhanced_statistics.py --env_name all --format latex
```

## Custom Experiments

### Adding New Systems
1. Create system directory: `systems/NewSystem/`
2. Add OpenDSS file: `systems/NewSystem/NewSystem.dss`
3. Register in `powergym/env_register.py`
4. Run experiments: `python run_exp.py --env NewSystem`

### Modifying Reward Function
Edit `powergym/env.py`:
```python
def _compute_reward(self, obs):
    # Custom reward logic
    voltage_penalty = ...
    switching_penalty = ...
    return voltage_penalty + switching_penalty
```

### Adding New Baselines
1. Create baseline script: `new_baseline.py`
2. Implement controller class
3. Add to `run_exp.py` evaluation loop

## Data and Results

### Directory Structure
All artifacts are written flat into `experiment_results/`:
```
experiment_results/
├── <method>_<env>_seed<seed>.zip / .pt   # Trained models (gitignored)
├── <method>_<env>_seed<seed>_training_data.json  # Training progress (gitignored)
├── results_<env>.json                    # Per-seed evaluation trajectories (tracked)
└── results_summary.csv                   # Aggregated mean ± std (tracked)
```

### Result Files
- `results_<env>.json`: Per-seed evaluation results used to regenerate all tables and figures
- `results_summary.csv`: Aggregated statistics across methods and seeds
- `opf_results_<env>.json` (repo root): OPF oracle evaluation results

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{dutta2025voltvar,
  title={Parameter-Shared Specialist Agents for Scalable Volt/Var Control in Power Distribution Networks},
  author={Dutta, Rochisnu and Swarup, K. Shanti},
  journal={IEEE Transactions on Power Systems},
  year={2025},
  note={Under review}
}
```
