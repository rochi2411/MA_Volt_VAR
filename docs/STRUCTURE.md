# Repository Structure

This document describes the organization of the MA_Volt_VAR repository.

## Current Structure (Flat Layout)

The repository currently uses a **flat structure** with all scripts in the root directory:

```
MA_Volt_VAR/
├── docs/                        # Documentation
│   ├── STRUCTURE.md            # This file
│   ├── INSTALLATION.md         # Detailed installation guide
│   └── EXPERIMENTS.md          # Experiment reproduction guide
│
├── powergym/                    # PowerGym environment
│   ├── env.py                  # Single-agent environment
│   ├── ma_env.py               # Multi-agent environment
│   ├── circuit.py              # Power system circuit interface
│   ├── env_register.py         # Environment registration
│   └── baselines.py            # Baseline implementations
│
├── systems/                     # Power system configurations
│   ├── 13Bus/                  # IEEE 13-Bus test feeder
│   ├── 34Bus/                  # IEEE 34-Bus test feeder
│   └── 123Bus/                 # IEEE 123-Bus test feeder
│
├── Training Scripts (root directory)
│   ├── train_marl.py           # IPPO specialist ensemble training
│   ├── train_monolithic.py     # Monolithic PPO baseline
│   ├── train_mappo.py          # MAPPO baseline
│   └── train_maddpg.py         # MADDPG baseline
│
├── Baseline Controllers (root directory)
│   ├── heuristic_agent.py      # Rule-based controller
│   └── opendss_auto_baseline.py # OpenDSS auto + OPF baselines
│
├── Analysis & Visualization (root directory)
│   ├── agent_behaviour_analysis.py  # Behavioral analysis
│   ├── enhanced_statistics.py       # Statistical analysis
│   └── aggregate_results.py         # Results aggregation
│
├── Experiment Runners (root directory)
│   ├── run_exp.py              # Main experiment runner
│   └── run_ablation.py         # Ablation studies
│
├── Model-Based OPF (root directory)
│   ├── opf_lindist3flow.py     # LinDist3Flow MILP oracle (HiGHS via CVXPY)
│   └── opf_results_*.json      # OPF oracle evaluation results (tracked)
│
├── Utilities (root directory)
│   ├── prepare_step_profiles.py # Load-profile preparation (10-min resolution)
│   ├── add_renewable_generation.py # Distributed-generation injection setup
│   ├── paper_style.py          # Shared matplotlib publication style
│   └── plot_profiles.py        # Input-profile figure generation
│
├── experiment_results/          # results_<sys>.json + results_summary.csv tracked;
│                                #   checkpoints (*.zip/*.pt) and training dumps gitignored
├── figures/                     # Publication input-profile figures (tracked)
├── analysis_plots/              # Behavioural / learning-curve plots (generated; gitignored)
├── venv/                        # Virtual environment (gitignored)
├── .gitignore                   # Git ignore rules
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── README.md                    # Main documentation
└── requirements.txt             # Python dependencies
```

## Proposed Structure (Organized Layout)

The proposed structure organizes the repository into clear categories:

```
MA_Volt_VAR/
├── docs/                           # Documentation
│   ├── STRUCTURE.md               # This file
│   ├── INSTALLATION.md            # Detailed installation guide
│   └── EXPERIMENTS.md             # Experiment reproduction guide
│
├── scripts/                        # Main executable scripts
│   ├── training/                  # Training scripts
│   │   ├── train_marl.py         # IPPO specialist ensemble training
│   │   ├── train_monolithic.py   # Monolithic PPO baseline
│   │   ├── train_mappo.py        # MAPPO baseline
│   │   └── train_maddpg.py       # MADDPG baseline
│   │
│   ├── baselines/                 # Baseline controllers
│   │   ├── heuristic_agent.py    # Rule-based controller
│   │   └── opendss_auto_baseline.py  # OpenDSS auto + OPF baselines
│   │
│   ├── analysis/                  # Analysis and visualization
│   │   ├── agent_behaviour_analysis.py  # Behavioral analysis
│   │   ├── enhanced_statistics.py       # Statistical analysis
│   │   ├── aggregate_results.py         # Results aggregation
│   │
│   └── experiments/               # Experiment runners
│       ├── run_exp.py            # Main experiment runner
│       └── run_ablation.py       # Ablation studies
│
├── utils/                         # Utility modules
│   ├── loadprofile.py            # Load profile generation
│   ├── prepare_step_profiles.py  # Step profile preparation
│   └── add_renewable_generation.py  # Renewable generation addition
│
├── powergym/                      # PowerGym environment
│   ├── env.py                    # Single-agent environment
│   ├── ma_env.py                 # Multi-agent environment
│   ├── circuit.py                # Power system circuit interface
│   ├── env_register.py           # Environment registration
│   └── baselines.py              # Baseline implementations
│
├── systems/                       # Power system configurations
│   ├── 13Bus/                    # IEEE 13-Bus test feeder
│   ├── 34Bus/                    # IEEE 34-Bus test feeder
│   └── 123Bus/                   # IEEE 123-Bus test feeder
│
├── results/                       # Experiment results (gitignored)
│   ├── models/                   # Trained models
│   ├── logs/                     # Training logs
│   └── metrics/                  # Evaluation metrics
│
│
├── tests/                         # Unit tests
│   └── test_env.py               # Environment tests
│
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
└── setup.py                       # Package setup (optional)
```

## File Categories (Current Structure)

All Python scripts are located in the **root directory**:

### Training Scripts
- **train_marl.py**: IPPO with parameter sharing (our main method)
- **train_monolithic.py**: Single-agent PPO baseline
- **train_mappo.py**: Multi-agent PPO with centralized critic
- **train_maddpg.py**: Multi-agent DDPG with Gumbel-Softmax

### Baseline Controllers
- **heuristic_agent.py**: Rule-based voltage control
- **opendss_auto_baseline.py**: OpenDSS auto-control and OPF

### Analysis Tools
- **agent_behaviour_analysis.py**: Comprehensive behavioral analysis
- **enhanced_statistics.py**: Statistical significance testing
- **aggregate_results.py**: Multi-seed result aggregation

### Experiment Runners
- **run_exp.py**: Complete multi-seed, multi-system experiments
- **run_ablation.py**: Ablation studies

### Utilities
- **prepare_step_profiles.py**: Prepare step-wise load profiles
- **add_renewable_generation.py**: Add PV/wind generation

## Why Flat Structure?

The current flat structure has advantages:
- ✅ **Simple imports**: All scripts can import from each other without complex path management
- ✅ **Easy navigation**: All main scripts visible at root level
- ✅ **Quick prototyping**: Fast to add new scripts and experiments

## Migration to Organized Structure (Optional)

If you prefer better organization, you can reorganize into subdirectories. This would provide:
- ✅ **Better organization**: Clear separation of concerns
- ✅ **Scalability**: Easier to manage as project grows
- ✅ **Professional structure**: Standard for larger projects


