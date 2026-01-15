import gymnasium as gym
import numpy as np
import powergym.env_register as env_register
from stable_baselines3 import PPO
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import re
import argparse

from train_monolithic import PowerGymWrapper # Re-use the monolithic wrapper
from heuristic_agent import HeuristicAgent # Re-use the heuristic agent class
from train_marl import IPPO_Wrapper # Re-use the IPPO wrapper

# --- Helper Functions ---
def calculate_metrics(env, obs_history, info_history):
    total_voltage_violation = 0
    total_power_loss = 0
    num_steps = len(obs_history)

    for i in range(num_steps):
        # Voltage Violation: Use the reward_func from powergym.Env
        pass

    return 0, 0 

def evaluate_agent(agent_name, agent_instance, env_wrapper_class, env_name, model_path=None, eval_episodes=5):
    print(f"\n--- Evaluating {agent_name} ---")
    
    # Initialize environment for this agent
    if agent_name == "Monolithic":
        env = env_wrapper_class(env_name) # PowerGymWrapper
    elif agent_name == "Specialist Ensemble":
        env = env_wrapper_class(env_name) # IPPO_Wrapper
    else: # Heuristic Agent
        env = env_wrapper_class(env_name, wrap_observation=False) # raw powergym.Env

    if model_path:
        if os.path.exists(model_path):
            model = PPO.load(model_path, env=env)
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: Model not found at {model_path}. Evaluation will fail/be random.")
            return {"avg_reward": -999, "avg_voltage_violation": 999, "avg_power_loss": 999, "case_study_data": None}

    episode_rewards = []
    episode_voltage_violations = []
    episode_power_losses = []
    
    # For case study
    case_study_data = None
    violation_found = False

    for ep in range(eval_episodes):
        # Reset environment
        if agent_name == "Monolithic" or agent_name == "Specialist Ensemble":
            obs, info = env.reset(seed=ep) # Gymnasium API
        else: # Heuristic
            obs = env.reset(load_profile_idx=ep) # Old gym API, no info
            
        current_episode_reward = 0
        episode_raw_obs_history = [] # Store raw observations for metric calculation
        
        # Get the actual raw_env for metric calculation
        if agent_name == "Monolithic":
            raw_env_for_metrics = env.env 
        elif agent_name == "Specialist Ensemble":
            raw_env_for_metrics = env.ma_env.raw_env
        else: # Heuristic
            raw_env_for_metrics = env # Heuristic agent directly uses raw_env
            
        # Initial raw obs to calculate metrics for step 0
        initial_raw_obs = raw_env_for_metrics.obs.copy()
        
        # Extract initial voltages and power loss
        initial_voltage_violation, initial_power_loss = get_metrics_from_raw_obs(raw_env_for_metrics, initial_raw_obs)
        
        # Store for this episode
        episode_raw_obs_history.append(initial_raw_obs)
        
        done = False
        step_idx = 0
        
        while not done:
            if agent_name == "Monolithic" or agent_name == "Specialist Ensemble":
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else: # Heuristic
                action = agent_instance.choose_action(obs)
                obs, reward, done, info = env.step(action) # Old gym API
            
            current_episode_reward += reward
            step_idx += 1
            
            # Store raw observations for current step (after action)
            current_raw_obs = raw_env_for_metrics.obs.copy()
            episode_raw_obs_history.append(current_raw_obs)

            # Check for violation for case study if not already found
            if not violation_found:
                current_v_vio, _ = get_metrics_from_raw_obs(raw_env_for_metrics, current_raw_obs)
                if current_v_vio > 0:
                    violation_found = True
                    case_study_data = {
                        "agent": agent_name,
                        "episode": ep,
                        "step": step_idx,
                        "raw_obs_at_violation": current_raw_obs,
                        "action_taken": action, # Action that led to this state
                        "ma_env_action_dict": env.action_buffer if agent_name == "Specialist Ensemble" else None # To get individual agent actions
                    }
                    
        episode_rewards.append(current_episode_reward)
        
        # Calculate episode-level metrics from history
        ep_v_vio, ep_p_loss = calculate_episode_metrics(raw_env_for_metrics, episode_raw_obs_history)
        episode_voltage_violations.append(ep_v_vio)
        episode_power_losses.append(ep_p_loss)

    avg_reward = np.mean(episode_rewards)
    avg_v_violation = np.mean(episode_voltage_violations)
    avg_p_loss = np.mean(episode_power_losses)
    
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Avg Voltage Violation: {avg_v_violation:.4f}")
    print(f"  Avg Power Loss: {avg_p_loss:.4f}")
    
    return {"avg_reward": avg_reward, "avg_voltage_violation": avg_v_violation, 
            "avg_power_loss": avg_p_loss, "case_study_data": case_study_data}

def get_metrics_from_raw_obs(raw_env, raw_obs):
    # Calculate voltage violation
    total_violation = 0
    for bus_name, voltages in raw_obs['bus_voltages'].items():
        max_v = max(voltages)
        min_v = min(voltages)
        total_violation += max(0, max_v - 1.05)
        total_violation += max(0, 0.95 - min_v)
    
    # Power loss is already in raw_obs
    power_loss = raw_obs['power_loss']
    
    return total_violation, power_loss

def calculate_episode_metrics(raw_env, episode_raw_obs_history):
    total_v_violation = 0
    total_p_loss = 0
    for raw_obs in episode_raw_obs_history:
        v_vio, p_loss = get_metrics_from_raw_obs(raw_env, raw_obs)
        total_v_violation += v_vio
        total_p_loss += p_loss
    
    # Average over steps
    return total_v_violation / len(episode_raw_obs_history), total_p_loss / len(episode_raw_obs_history)

def plot_training_curves(log_dirs, agent_names, plot_dir):
    plt.figure(figsize=(10, 6))
    for log_dir, name in zip(log_dirs, agent_names):
        try:
            df = pd.read_csv(os.path.join(log_dir, 'monitor.csv'), skiprows=1) # Skip initial comment line
            # Extract episode rewards (r) and episode lengths (l)
            # 'l' is episode length, 'r' is episode reward, 't' is time elapsed
            # Each row represents a completed episode
            plt.plot(df['t'], df['r'], label=f'{name} Reward')
        except FileNotFoundError:
            print(f"Warning: Monitor log not found for {name} at {log_dir}")
            continue
    
    plt.title('Training Rewards Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    # Save the plot
    file_path = os.path.join(plot_dir, 'training_rewards.png')
    with open(file_path, 'wb') as f:
        plt.savefig(f, format='png')
    plt.close()

def plot_comparative_analysis(metrics_data, plot_dir):
    labels = list(metrics_data.keys())
    rewards = [data["avg_reward"] for data in metrics_data.values()]
    v_violations = [data["avg_voltage_violation"] for data in metrics_data.values()]
    p_losses = [data["avg_power_loss"] for data in metrics_data.values()]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, rewards, width, label='Avg Reward')
    rects2 = ax.bar(x, v_violations, width, label='Avg Voltage Violation')
    rects3 = ax.bar(x + width, p_losses, width, label='Avg Power Loss')

    ax.set_ylabel('Metrics Value')
    ax.set_title('Comparative Analysis of Agents')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    # Save the plot
    file_path = os.path.join(plot_dir, 'comparative_metrics.png')
    with open(file_path, 'wb') as f:
        plt.savefig(f, format='png')
    plt.close()

def plot_case_study(case_study_data, plot_dir, env_name):
    if not case_study_data:
        print("No case study data available or no violation found.")
        return

    agent_name = case_study_data['agent']
    episode = case_study_data['episode']
    step = case_study_data['step']
    raw_obs = case_study_data['raw_obs_at_violation']
    action_taken = case_study_data['action_taken'] # This is the action from the agent wrapper
    ma_env_action_dict = case_study_data['ma_env_action_dict'] # Individual MA actions

    print(f"\n--- Case Study: {agent_name}, Episode {episode}, Step {step} ---")
    print(f"  Action taken by {agent_name} (wrapper output): {action_taken}")
    if ma_env_action_dict:
        print("  Individual Agent Actions:")
        for k, v in ma_env_action_dict.items():
            print(f"    {k}: {v}")
    
    # Recreate the raw environment to use its plotting capabilities
    if agent_name == "Monolithic":
        # Need to create an Env instance, then reset it and get obs to match
        temp_env = env_register.make_env(env_name, wrap_observation=False)
        temp_env.reset(load_profile_idx=episode) # Need to set correct load profile
        # Now, manually set the state of temp_env to raw_obs
        temp_env.obs = raw_obs
    elif agent_name == "Specialist Ensemble":
        temp_ma_env = MultiAgentPowerGrid(env_name)
        temp_ma_env.raw_env.reset(load_profile_idx=episode)
        temp_ma_env.raw_env.obs = raw_obs # Directly set obs
        temp_env = temp_ma_env.raw_env
    else: # Heuristic
        temp_env = env_register.make_env(env_name, wrap_observation=False)
        temp_env.reset(load_profile_idx=episode) # Set correct load profile
        temp_env.obs = raw_obs # Directly set obs

    # Check actual voltage violation for this raw_obs
    v_vio, p_loss = get_metrics_from_raw_obs(temp_env, raw_obs)
    print(f"  Voltage Violation at this step: {v_vio:.4f}")
    print(f"  Power Loss at this step: {p_loss:.4f}")

    # Plot the voltage profile
    try:
        fig, _ = temp_env.plot_graph()
        plt.title(f'Voltage Profile: {agent_name} - Episode {episode}, Step {step} (Violation)')
        plt.tight_layout()
        # Save the plot
        file_path = os.path.join(plot_dir, f'case_study_{agent_name.replace(" ", "_")}_ep{episode}_step{step}.png')
        with open(file_path, 'wb') as f:
            plt.savefig(f, format='png')
        plt.close()
    except Exception as e:
        print(f"Failed to plot graph (likely missing layout info for {env_name}): {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Agents')
    parser.add_argument('--env_name', type=str, default='13Bus', help='Environment name (e.g., 13Bus, 34Bus)')
    args = parser.parse_args()
    
    ENV_NAME = args.env_name
    PLOT_DIR = f'evaluation_plots_{ENV_NAME}'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    EVAL_EPISODES = 5

    # --- Agent Setup ---
    # Assume generic model names based on env_name or specific files if available
    # For 34Bus/123Bus we expect models named accordingly if we run training script with args
    # train_monolithic saves as "monolithic_agent".
    # I should update train_monolithic.py to save with env_name.
    # I will modify evaluate_agents.py to look for "monolithic_agent_{ENV_NAME}.zip"
    
    monolithic_model_path = f"monolithic_agent_{ENV_NAME}.zip"
    specialist_ensemble_model_path = f"specialist_ensemble_{ENV_NAME}.zip"

    heuristic_agent_instance = HeuristicAgent(env_register.make_env(ENV_NAME, wrap_observation=False))

    results = {}
    case_studies = {}

    # --- Evaluate Monolithic Agent ---
    monolithic_results = evaluate_agent("Monolithic", None, PowerGymWrapper, ENV_NAME, monolithic_model_path, EVAL_EPISODES)
    results["Monolithic"] = monolithic_results
    case_studies["Monolithic"] = monolithic_results["case_study_data"]

    # --- Evaluate Specialist Ensemble Agent ---
    specialist_ensemble_results = evaluate_agent("Specialist Ensemble", None, IPPO_Wrapper, ENV_NAME, specialist_ensemble_model_path, EVAL_EPISODES)
    results["Specialist Ensemble"] = specialist_ensemble_results
    case_studies["Specialist Ensemble"] = specialist_ensemble_results["case_study_data"]

    # --- Evaluate Rule-based Heuristic Agent ---
    heuristic_results = evaluate_agent("Heuristic", heuristic_agent_instance, env_register.make_env, ENV_NAME, None, EVAL_EPISODES)
    results["Heuristic"] = heuristic_results
    case_studies["Heuristic"] = heuristic_results["case_study_data"]

    # --- Plotting ---
    print("\n--- Generating Plots ---")
    
    # Training Curves - Need to find logs.
    # train_monolithic logs to "logs/monolithic/monitor.csv" (hardcoded)
    # train_marl logs to "logs/specialist_ensemble/monitor.csv" (hardcoded)
    # I should update training scripts to log to env-specific folders.
    
    # Assuming training scripts are updated:
    log_dirs = [f"logs/monolithic_{ENV_NAME}/", f"logs/specialist_ensemble_{ENV_NAME}/"]
    
    plot_training_curves(
        log_dirs=log_dirs,
        agent_names=["Monolithic", "Specialist Ensemble"],
        plot_dir=PLOT_DIR
    )

    # Comparative Bar Charts
    plot_comparative_analysis(results, PLOT_DIR)

    # Case Studies
    for agent_name, data in case_studies.items():
        if data:
            plot_case_study(data, PLOT_DIR, ENV_NAME)

    print(f"\nEvaluation complete. Plots saved to '{PLOT_DIR}/' directory.")