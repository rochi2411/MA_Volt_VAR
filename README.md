# Multi-Agent Deep Reinforcement Learning for Volt-VAR Control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PowerGym](https://img.shields.io/badge/Environment-PowerGym-green.svg)](https://github.com/siemens/powergym)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org) -->

> **Specialist Ensemble Multi-Agent Reinforcement Learning for Autonomous Volt-VAR Control in Power Distribution Networks**

This repository implements a novel **Specialist Ensemble** approach using Independent PPO (IPPO) with parameter sharing for autonomous voltage regulation in distribution grids. By decomposing the complex joint action space into device-specific specialist agents (capacitors, voltage regulators, and battery energy storage systems), our method achieves superior scalability and robustness compared to monolithic and centralized multi-agent baselines.

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Key Features

- **Multi-Agent Decomposition**: Specialist agents for each device type (capacitors, regulators, batteries)
- **Parameter Sharing**: Efficient learning through shared policy networks with agent-type embeddings
- **Scalable Architecture**: Linear scaling with number of controllable devices
- **Robust Performance**: Maintains voltage profiles under observation uncertainty
- **Comprehensive Baselines**: Includes Monolithic PPO and rule-based Heuristic controllers

### Performance Comparison

Mean ± std over 5 seeds. **Bold** marks the best *deployable* method (the OPF oracle is an offline,
perfect-information upper bound, not a deployable controller — see note below).

**IEEE 13-Bus System**

| Method | Reward (↑) | Violation Index (↓) |
|--------|------------|---------------------|
| Specialist (Ours) | -2.98 ± 0.31 | 0.0203 ± 0.0178 |
| Monolithic PPO | -60.54 ± 19.48 | 0.0473 ± 0.0597 |
| MAPPO | -2.57 ± 0.06 | 0.0020 ± 0.0016 |
| **MADDPG** | **-2.56 ± 0.03** | 0.0002 ± 0.0004 |
| Heuristic (Rule-Based) | -29.33 ± 0.00 | 0.0005 ± 0.0000 |
| OpenDSS Auto + Droop | -23.97 ± 0.51 | **0.0000 ± 0.0000** |
| Model-based OPF (oracle)† | -26.59 ± 0.89 | 0.0026 ± 0.0015 |

**IEEE 34-Bus System**

| Method | Reward (↑) | Violation Index (↓) |
|--------|------------|---------------------|
| **Specialist (Ours)** | **-15.05 ± 4.31** | 0.0103 ± 0.0041 |
| Monolithic PPO | -96.87 ± 48.84 | 0.4656 ± 0.3494 |
| MAPPO | -20.20 ± 18.76 | 0.0204 ± 0.0303 |
| MADDPG | -23.49 ± 7.29 | 0.1623 ± 0.1985 |
| Heuristic (Rule-Based) | -79.15 ± 0.00 | 0.2763 ± 0.0000 |
| OpenDSS Auto + Droop | -25.14 ± 0.18 | **0.0017 ± 0.0000** |
| Model-based OPF (oracle)† | -14.09 ± 0.47 | 0.0077 ± 0.0009 |

**IEEE 123-Bus System**

| Method | Reward (↑) | Violation Index (↓) |
|--------|------------|---------------------|
| **Specialist (Ours)** | **-3.60 ± 0.99** | 0.0035 ± 0.0066 |
| Monolithic PPO | -170.25 ± 75.26 | 0.2697 ± 0.1558 |
| MAPPO | -3.65 ± 1.77 | 0.0013 ± 0.0016 |
| MADDPG | -3.66 ± 1.16 | 0.0549 ± 0.0674 |
| Heuristic (Rule-Based) | -30.93 ± 0.00 | 0.0000 ± 0.0000 |
| OpenDSS Auto + Droop | -27.69 ± 0.09 | **0.0000 ± 0.0000** |
| Model-based OPF (oracle)† | -23.69 ± 0.11 | **0.0000 ± 0.0000** |

> † **Model-based OPF (oracle)** is a multi-period LinDist3Flow MILP solved with full-horizon
> foresight and global state, then re-validated in OpenDSS on the true reward. It is an *offline upper
> bound*, not a deployable controller, and is reported only as a reference ceiling. The implementation
> is in [`opf_lindist3flow.py`](opf_lindist3flow.py).

*Aggregated from per-seed evaluation under `experiment_results/results_<system>.json`.*

### Key Findings

✅ **Best deployable method** on the 34-Bus (−15.05) and 123-Bus (−3.60); near-best on 13-Bus (within 0.42 of MADDPG)  
✅ **Approaches the perfect-information OPF oracle** on 34-Bus (−15.05 vs. −14.09, statistically indistinguishable) using only *local* observations and no forecast  
✅ **Massively outperforms Monolithic PPO** (81–98% higher reward), whose single joint policy fails to scale  
✅ **Lower evaluation variance than MAPPO** (4.3× on 34-Bus), with no catastrophic training crashes  
✅ **Scalable architecture**: parameter sharing keeps the policy size constant as the number of devices grows

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Policy Network πθ                  │
│                  (MLP with agent-type embedding)             │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Capacitor    │  │   Regulator   │  │    Battery    │
│  Specialist   │  │   Specialist  │  │   Specialist  │
│  (Binary)     │  │  (Discrete)   │  │ (Continuous)  │
└───────────────┘  └───────────────┘  └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│              PowerGym Distribution Network                   │
│            (IEEE 13/34/123 Bus Test Feeders)                │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
MA_Volt_VAR/
├── docs/                        # Documentation
│   ├── INSTALLATION.md         # Detailed install guide
│   ├── EXPERIMENTS.md          # Experiment reproduction
│   └── STRUCTURE.md            # Repo organization
│
├── powergym/                    # PowerGym environment
│   ├── env.py                  # Single-agent environment
│   ├── ma_env.py               # Multi-agent wrapper
│   ├── circuit.py              # OpenDSS interface
│   └── env_register.py         # Environment registration
│
├── systems/                     # IEEE test feeders
│   ├── 13Bus/                  # IEEE 13-Bus
│   ├── 34Bus/                  # IEEE 34-Bus
│   └── 123Bus/                 # IEEE 123-Bus
│
├── Training Scripts (root directory)
│   ├── train_marl.py           # Specialist ensemble (IPPO)
│   ├── train_monolithic.py     # Monolithic PPO
│   ├── train_mappo.py          # MAPPO baseline
│   └── train_maddpg.py         # MADDPG baseline
│
├── Baseline Controllers (root directory)
│   ├── heuristic_agent.py      # Rule-based controller
│   └── opendss_auto_baseline.py # OpenDSS + OPF
│
├── Analysis & Visualization (root directory)
│   ├── agent_behaviour_analysis.py  # Behavioral analysis
│   ├── enhanced_statistics.py       # Statistical tests
│   └── aggregate_results.py         # Result aggregation
│
├── Experiment Runners (root directory)
│   ├── run_exp.py              # Complete experiments (includes plotting)
│   └── run_ablation.py         # Ablation studies
│
├── Utilities (root directory)
│   ├── prepare_step_profiles.py # Load-profile preparation (10-min resolution)
│   └── add_renewable_generation.py # Distributed-generation injection setup
│
├── experiment_results/          # Per-seed eval JSONs tracked; checkpoints/dumps gitignored
├── figures/                     # Publication input-profile figures (tracked)
├── opf_*.json                   # OPF oracle evaluation results (tracked)
├── analysis_plots/              # Behavioural/learning plots (generated; gitignored)
├── venv/                        # Virtual environment (gitignored)
├── requirements.txt             # Dependencies
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
└── README.md                    # This file
```

**Note**: Repository uses a flat structure with all scripts in the root directory. See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for a proposed organized structure.

## 🚀 Quick Start

### 30-Second Demo

```bash
# Clone and setup
git clone https://github.com/rochi2411/MA_Volt_VAR.git
cd MA_Volt_VAR
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train specialist ensemble (15 min on CPU)
python train_marl.py --env_name 13Bus --steps 50000 --seed 42

# Visualize results
python agent_behaviour_analysis.py --env_name 13Bus
```

**Output**: Trained model + 8 publication-quality plots in `analysis_plots/`

---

## 📥 Installation

### Prerequisites
- **Python**: 3.9 or higher
- **OpenDSS**: Power flow simulation engine
- **OS**: Linux, macOS, or Windows (WSL2 recommended)

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/rochi2411/MA_Volt_VAR.git
cd MA_Volt_VAR

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "from powergym.env_register import make_env; print('✓ Installation successful!')"
```

**Detailed installation guide**: See [`docs/INSTALLATION.md`](docs/INSTALLATION.md)

---

## 🎮 Usage

### Training Individual Methods

#### 1. Specialist Ensemble (IPPO) - Our Method
```bash
python train_marl.py --env_name 13Bus --steps 50000 --seed 42
```

**Arguments**:
- `--env_name`: System to train on (`13Bus`, `34Bus`, `123Bus`)
- `--steps`: Total training timesteps (default: 50000)
- `--seed`: Random seed for reproducibility
- `--model_name`: Custom model name (optional)
- `--verbose`: Verbosity level (0=silent, 1=progress)
- `--no_save_data`: Disable training data saving

**Note**: Hyperparameters (learning rate, batch size, network architecture) are hardcoded in the training scripts. To modify them, edit the source files directly.

#### 2. Monolithic PPO Baseline
```bash
python train_monolithic.py --env_name 13Bus --steps 50000 --seed 42
```

#### 3. MAPPO Baseline (Centralized Critic)
```bash
python train_mappo.py --env_name 13Bus --steps 50000 --seed 42
```

#### 4. MADDPG Baseline
```bash
python train_maddpg.py --env_name 13Bus --steps 50000 --seed 42
```

### Running Complete Experiments

```bash
# All methods, all systems, 5 seeds (paper results)
python run_exp.py --env all --seeds 5

# Single system, 10 seeds (extended validation)
python run_exp.py --env 13Bus --seeds 10

# Skip training, only evaluate baselines
python run_exp.py --env all --seeds 5 --skip_training
```

### Analysis and Visualization

```bash
# Generate all plots for one system
python agent_behaviour_analysis.py --env_name 13Bus

# All systems
python agent_behaviour_analysis.py --env_name all

# Include robustness analysis (observation noise)
python agent_behaviour_analysis.py --env_name 13Bus --robustness

# Aggregate multi-seed results
python aggregate_results.py --results_dir ./experiment_results
```

**Detailed usage guide**: See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

### Generated Outputs

**Trained Models**: written flat into `experiment_results/` (gitignored — regenerated by training)
- `marl_<env>_seed<seed>.zip` (Specialist / IPPO)
- `monolithic_<env>_seed<seed>.zip`
- `mappo_<env>_seed<seed>.zip`
- `maddpg_<env>_seed<seed>.pt`

**Plots**: `analysis_plots/` (8 plots per system)
- `regulator_comparison_*.png` - Tap position trajectories
- `capacitor_comparison_*.png` - Switching patterns
- `battery_comparison_*.png` - Battery dispatch
- `voltage_profile_*.png` - Voltage over time
- `cumulative_rewards_*.png` - Reward comparison
- `violation_heatmap_*.png` - Violation patterns
- `training_curve_*.png` - Learning curves
- `robustness_*.png` - Noise robustness

**Metrics**: `experiment_results/`
- `results_<env>.json` — per-seed evaluation trajectories (committed)
- `results_summary.csv` — aggregated mean ± std across methods

---

## 📊 Results

- **Headline numbers**: see [Performance Comparison](#performance-comparison) above (5-seed mean ± std).
- **Per-seed evaluation data**: `experiment_results/results_<system>.json` (committed; used to regenerate every table and figure).
- **OPF oracle reference**: `opf_results_<system>.json`.
- **Input-profile figures** (load, solar irradiance, combined): committed under [`figures/`](figures/).
- **Behavioural / learning-curve plots**: regenerated locally into `analysis_plots/` via the analysis scripts (gitignored; reproducible — see [Usage](#-usage)).

---

## ⚙️ Configuration

### System Specifications

| Parameter | 13-Bus | 34-Bus | 123-Bus |
|-----------|--------|--------|---------|
| Capacitors | 2 | 2 | 4 |
| Regulators | 3 | 6 | 7 |
| Batteries | 1 | 2 | 4 |
| Total Agents | 6 | 10 | 15 |
| Episode Length | 144 steps | 144 steps | 144 steps |

*Each episode simulates a 24-hour horizon at 10-minute resolution (144 timesteps) with time-varying stochastic load profiles and distributed-generation injections.*

### Training Hyperparameters

```python
# Specialist Ensemble (IPPO) - Default Config
config = {
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
```

**To modify hyperparameters**: Edit the training script source code directly (around line 289-310 in `train_marl.py`).

---

## 📚 Documentation

### Guides
- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Experiment Guide](docs/EXPERIMENTS.md)**: Reproduce paper results
- **[Structure Guide](docs/STRUCTURE.md)**: Repository organization

### Method Details

#### Specialist Ensemble (IPPO with Parameter Sharing)

Our approach uses **Independent PPO (IPPO)** where all specialist agents share a single policy network with agent-type embeddings:

**Key Advantages**:
- ✅ **Sample Efficiency**: Shared experiences across agents
- ✅ **Scalability**: O(1) parameters vs. O(n²) for centralized
- ✅ **Generalization**: Transfer learning via embeddings
- ✅ **Robustness**: Decentralized execution

#### Baselines

1. **Monolithic PPO**: Single agent controlling the full joint action space
2. **MAPPO**: Centralized critic, decentralized actors
3. **MADDPG**: Centralized critics with Gumbel-Softmax discrete actions
4. **Heuristic**: Rule-based local voltage control
5. **OpenDSS Auto**: Built-in regulator/capacitor droop control
6. **Model-based OPF (oracle)**: Multi-period LinDist3Flow MILP solved with full-horizon foresight (perfect-information offline upper bound, re-validated in OpenDSS)

<!-- ---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{dutta2025voltvar,
  title={Parameter-Shared Specialist Agents for Scalable Volt/Var Control in Power Distribution Networks},
  author={Dutta, Rochisnu and Swarup, K. Shanti},
  journal={IEEE Transactions on Power Systems},
  year={2025},
  note={Under review}
}
``` -->

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon:
- **[PowerGym](https://github.com/siemens/powergym)**: RL environment for power systems
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**: RL algorithm implementations  
- **[OpenDSS](https://www.epri.com/pages/sa/opendss)**: Power flow simulation engine
- **IEEE Test Feeders**: Standard distribution system benchmarks

---

## 📧 Contact

**Rochisnu Dutta**  
🐙 GitHub: [@rochi2411](https://github.com/rochi2411)  
🔗 Project: [MA_Volt_VAR](https://github.com/rochi2411/MA_Volt_VAR)

---

## ⭐ Star History

If you find this work useful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=rochi2411/MA_Volt_VAR&type=Date)](https://star-history.com/#rochi2411/MA_Volt_VAR&Date)