# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Rule-based agent for voltage regulation
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
import os
import argparse
import random
import itertools

from powergym.env_register import make_env, remove_parallel_dss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                         help='random seed')
    parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                         help='maximum number of steps')
    parser.add_argument('--use_plot', type=lambda x: str(x).lower()=='true', default=False)
    args = parser.parse_args()
    return args

class HeuristicAgent:
    def __init__(self, env):
        self.env = env
        self.cap_num = env.cap_num
        self.reg_num = env.reg_num
        self.bat_num = env.bat_num
        self.reg_act_num = env.reg_act_num
        self.bat_act_num = env.bat_act_num

    def choose_action(self, obs):
        # Default to all capacitors off
        capacitor_actions = [0] * self.cap_num

        # Check bus voltages
        # obs['bus_voltages'] is a dict where values are lists of phase voltages
        voltages = []
        for bus_name in obs['bus_voltages']:
            voltages.extend(obs['bus_voltages'][bus_name])
        
        # Rule: if any voltage > 1.05, switch all capacitors OFF; else if any voltage < 0.95, switch all capacitors ON.
        # Otherwise, keep them in the default OFF state.
        if any(v > 1.05 for v in voltages):
            capacitor_actions = [0] * self.cap_num # Switch OFF
        elif any(v < 0.95 for v in voltages):
            capacitor_actions = [1] * self.cap_num # Switch ON
        else:
            capacitor_actions = [0] * self.cap_num # Default OFF if no violation
        
        # For regulators and batteries, use dummy actions for now
        # Regulators: typically 0-indexed tap positions, so 0 or env.reg_act_num // 2
        regulator_actions = [0] * self.reg_num 
        # Batteries: 0 if discrete, or 0.0 if continuous.
        battery_actions = [0.0 if self.bat_act_num == np.inf else 0] * self.bat_num

        # Combine all actions
        action = np.array(capacitor_actions + regulator_actions + battery_actions)
        return action

def run_heuristic_agent(args, worker_idx=None, use_plot=False):
    cwd = os.getcwd()
    
    env = make_env(args.env_name, worker_idx=worker_idx, wrap_observation=False)
    env.seed(args.seed + (worker_idx if worker_idx is not None else 0))

    print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num, env.bat_num))
    print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
    print('-'*80)

    agent = HeuristicAgent(env)

    obs = env.reset(load_profile_idx=0)
    
    if use_plot and not os.path.exists(os.path.join(cwd,'heuristic_agent_plots')):
        os.makedirs(os.path.join(cwd,'heuristic_agent_plots'))

    episode_reward = 0.0
    for i in range(env.horizon):
        action = agent.choose_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        print(f"Step: {i}, Reward: {reward}, Action: {action[:env.cap_num]} (Capacitor actions)")

        if use_plot:
            fig, _ = env.plot_graph()
            fig.tight_layout(pad=0.1)
            fig.savefig(os.path.join(cwd,'heuristic_agent_plots', f'node_voltage_{str(i).zfill(4)}.png'))
            plt.close()
            
        if done:
            break
    
    print(f'Episode finished. Total reward: {episode_reward}')

    if use_plot:
        fig, _ = env.plot_graph(show_voltages=False)
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(cwd, 'heuristic_agent_plots', 'system_layout.pdf'))

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_heuristic_agent(args)

