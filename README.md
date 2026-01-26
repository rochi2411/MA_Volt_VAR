# Multi-Agent Deep Reinforcement Learning for Volt-VAR Control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PowerGym](https://img.shields.io/badge/Environment-PowerGym-green.svg)](https://github.com/siemens/powergym)

A **Specialist Ensemble** approach using Independent PPO (IPPO) with parameter sharing for autonomous Volt-VAR control in power distribution networks. This framework decomposes the complex joint action space into device-specific specialist agents (capacitors, voltage regulators, and battery energy storage systems), enabling scalable and robust voltage regulation.

## 🎯 Key Features

- **Multi-Agent Decomposition**: Specialist agents for each device type (capacitors, regulators, batteries)
- **Parameter Sharing**: Efficient learning through shared policy networks with agent-type embeddings
- **Scalable Architecture**: Linear scaling with number of controllable devices
- **Robust Performance**: Maintains voltage profiles under observation uncertainty
- **Comprehensive Baselines**: Includes Monolithic PPO and rule-based Heuristic controllers

## 📊 Results Summary

| System | Method | Reward (↑) | Violation Index (↓) | p-value |
|--------|--------|------------|---------------------|---------|
| **IEEE 13-Bus** | Specialist | **-0.73 ± 0.24** | **0.003 ± 0.005** | — |
| | Monolithic | -10.95 ± 4.01 | 0.083 ± 0.063 | p=0.007** |
| | Heuristic | -6.31 ± 0.00 | 0.003 ± 0.000 | p<0.001*** |
| **IEEE 34-Bus** | Specialist | **-5.39 ± 3.04** | **0.023 ± 0.016** | — |
| | Monolithic | -21.37 ± 5.14 | 0.285 ± 0.164 | p=0.001** |
| | Heuristic | -67.55 ± 0.00 | 1.819 ± 0.000 | p<0.001*** |
| **IEEE 123-Bus** | Specialist | **-0.90 ± 0.18** | **0.004 ± 0.008** | — |
| | Monolithic | -13.31 ± 1.30 | 0.019 ± 0.033 | p<0.001*** |
| | Heuristic | -12.57 ± 0.00 | 0.000 ± 0.000 | p<0.001*** |

*Results averaged over 5 random seeds with statistical significance testing.*

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
├── powergym/                    # PowerGym environment (submodule)
├── systems/                     # System-specific configurations
├── agent_behaviour_analysis.py  # Behavioral analysis & visualization
├── heuristic_agent.py          # Rule-based baseline controller
├── run_exp.py                  # Multi-seed experiment runner
├── train_marl.py               # IPPO training with parameter sharing
├── train_monolithic.py         # Single-agent PPO baseline
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenDSS (for power flow simulation)

### Installation

```bash
# Clone the repository
git clone https://github.com/rochi2411/MA_Volt_VAR.git
cd MA_Volt_VAR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train Specialist Ensemble (IPPO) on IEEE 13-Bus
python train_marl.py --env_name 13Bus --steps 50000 --seed 42

# Train Monolithic PPO baseline
python train_monolithic.py --env_name 13Bus --steps 50000 --seed 42

# Run complete experiments (5 seeds × 3 systems × 3 methods)
python run_exp.py --env_name all --seeds 42,123,456,789,1011
```

### Evaluation & Analysis

```bash
# Generate behavioral analysis plots
python agent_behaviour_analysis.py --env_name 13Bus --robustness

# Run on all systems
python agent_behaviour_analysis.py --env_name all
```

## 📈 Generated Visualizations

The analysis script generates publication-quality plots:

| Plot | Description |
|------|-------------|
| `regulator_comparison_*.png` | Tap position trajectories for all methods |
| `capacitor_comparison_*.png` | Capacitor switching patterns |
| `battery_comparison_*.png` | Battery dispatch strategies |
| `voltage_profile_*.png` | Min/Mean/Max voltage over time |
| `cumulative_rewards_*.png` | Reward accumulation comparison |
| `violation_heatmap_*.png` | Temporal violation patterns |
| `training_curve_*.png` | Learning curves (5 seeds) |
| `robustness_*.png` | Performance under observation noise |

## 🔧 Configuration

### Environment Parameters

| Parameter | 13-Bus | 34-Bus | 123-Bus |
|-----------|--------|--------|---------|
| Capacitors | 2 | 2 | 4 |
| Regulators | 3 | 6 | 7 |
| Batteries | 1 | 2 | 4 |
| Total Agents | 6 | 10 | 15 |
| Episode Length | 24 steps | 24 steps | 24 steps |

### Training Hyperparameters

```python
# Default PPO hyperparameters (Stable-Baselines3)
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
```

## 📚 Method Details

### Specialist Ensemble (IPPO with Parameter Sharing)

Our approach uses **Independent PPO (IPPO)** where all specialist agents share a single policy network. To enable the shared network to distinguish between agent types, we augment observations with agent-type identifiers.

**Key advantages:**
- **Sample Efficiency**: Shared experiences across agents
- **Scalability**: O(1) policy parameters regardless of agent count
- **Generalization**: Agent-type embeddings enable transfer learning

### Heuristic Baseline (Algorithm 1)

Rule-based controller implementing standard utility practices:

```
For each timestep:
  1. Capacitors: ON if V_min < 0.95, OFF if V_max > 1.05
  2. Regulators: Tap adjustment with deadband (±0.00625 p.u.)
  3. Batteries: Discharge if V_min < 0.95, Charge if V_max > 1.05
```

<!-- ## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{dutta2025marl_voltvar,
  title={Multi-Agent Deep Reinforcement Learning for Autonomous Volt-VAR Control in Distribution Networks},
  author={Dutta, Rochisnu},
  journal={arXiv preprint},
  year={2025}
}
``` -->

<!-- ## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PowerGym](https://github.com/siemens/powergym) - Reinforcement learning environment for power systems
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithm implementations
- [OpenDSS](https://www.epri.com/pages/sa/opendss) - Power system simulation engine

## 📧 Contact

**Rochisnu Dutta** - [GitHub](https://github.com/rochi2411)

Project Link: [https://github.com/rochi2411/MA_Volt_VAR](https://github.com/rochi2411/MA_Volt_VAR)