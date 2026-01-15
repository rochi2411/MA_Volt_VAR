# PowerGym Control Agents

This repository contains implementations of various Reinforcement Learning (RL) agents for the [PowerGym](https://github.com/siemens/powergym) environment, specifically designed to control capacitors, regulators, and batteries in power distribution networks (IEEE 13, 34, and 123 bus systems).

## Supported Agents

1.  **Monolithic DRL Agent (`train_monolithic.py`)**: A centralized PPO agent that controls all devices in the grid simultaneously.
2.  **Specialist Ensemble DRL Agent (`train_marl.py`)**: A multi-agent system (using Independent PPO with parameter sharing) where agents optimize local objectives while contributing to global stability.
3.  **Heuristic Agent (`heuristic_agent.py`)**: A rule-based baseline that switches capacitors based on voltage thresholds (< 0.95 pu ON, > 1.05 pu OFF).

## Installation

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train Agents

You can train agents on different environments (`13Bus`, `34Bus`, `123Bus`).

**Train Monolithic Agent:**
```bash
python train_monolithic.py --env_name 13Bus --steps 5000
```

**Train Specialist Ensemble Agent:**
```bash
python train_marl.py --env_name 13Bus --steps 5000
```

### 2. Evaluate Agents

After training, evaluate the performance of all agents (Monolithic, Specialist, and Heuristic) and generate comparative metrics.

```bash
python evaluate_agents.py --env_name 13Bus
```

This script will:
*   Load the trained models (e.g., `monolithic_agent_13Bus.zip`).
*   Run evaluation episodes.
*   Print average Reward, Voltage Violation, and Power Loss.

## Training Results & Curves

Training curves showing the learning progress (Reward vs. Time) for both DRL agents across all three systems have been generated:

*   **13Bus**: `training_curve_13Bus.png`
*   **34Bus**: `training_curve_34Bus.png`
*   **123Bus**: `training_curve_123Bus.png`

**Summary of Findings:**
The **Specialist Ensemble** consistently outperforms the Monolithic agent, especially in larger systems (34Bus, 123Bus), demonstrating faster convergence and higher final rewards. The Monolithic agent often struggles to improve beyond a suboptimal baseline in high-dimensional state spaces.

## Results Summary (Sample)

| Environment | Best Agent | Key Metric |
| :--- | :--- | :--- |
| **13Bus** | Specialist | Lowest Voltage Violation (0.0257 vs 0.39) |
| **34Bus** | Specialist | Lowest Voltage Violation (0.95 vs 1.30) |
| **123Bus** | Specialist | Lowest Voltage Violation (0.85 vs 2.72) |
