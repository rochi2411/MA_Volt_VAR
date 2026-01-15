
import gym
import numpy as np
from powergym.env import Env

class MultiAgentPowerGrid:
    """
    A Multi-Agent Wrapper for PowerGym.
    
    Splits the centralized environment into distinct agents:
    - Capacitors
    - Regulators
    - Batteries
    
    Each agent observes only its LOCAL state (voltage at connection bus + internal status).
    """
    def __init__(self, env_name, worker_idx=None):
        # Initialize the standard centralized environment
        from powergym.env_register import make_env
        self.raw_env = make_env(env_name, worker_idx=worker_idx, wrap_observation=False)
        
        # Do not flatten observations; we need the raw dict to slice it
        self.raw_env.reset_obs_space()
        
        self.agents = []
        self.agent_types = {} # Map agent_id -> 'cap', 'reg', 'bat'
        self.agent_mapping = {} # Map agent_id -> circuit component name
        
        # 1. Register Agents
        # Capacitors
        for name in self.raw_env.cap_names:
            agent_id = f"cap_{name}"
            self.agents.append(agent_id)
            self.agent_types[agent_id] = 'cap'
            self.agent_mapping[agent_id] = name
            
        # Regulators
        for name in self.raw_env.reg_names:
            agent_id = f"reg_{name}"
            self.agents.append(agent_id)
            self.agent_types[agent_id] = 'reg'
            self.agent_mapping[agent_id] = name
            
        # Batteries
        for name in self.raw_env.bat_names:
            agent_id = f"bat_{name}"
            self.agents.append(agent_id)
            self.agent_types[agent_id] = 'bat'
            self.agent_mapping[agent_id] = name

        # 2. Define Action Spaces
        self.action_spaces = {}
        for agent_id in self.agents:
            atype = self.agent_types[agent_id]
            if atype == 'cap':
                self.action_spaces[agent_id] = gym.spaces.Discrete(2)
            elif atype == 'reg':
                self.action_spaces[agent_id] = gym.spaces.Discrete(self.raw_env.reg_act_num)
            elif atype == 'bat':
                if self.raw_env.bat_act_num == float('inf'):
                    self.action_spaces[agent_id] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
                else:
                    self.action_spaces[agent_id] = gym.spaces.Discrete(self.raw_env.bat_act_num)

    def reset(self):
        """
        Returns:
            obs_dict: {agent_id: local_observation}
        """
        global_obs = self.raw_env.reset()
        return self._extract_local_obs(global_obs)

    def step(self, action_dict):
        """
        Args:
            action_dict: {agent_id: action}
        
        Returns:
            obs_dict, reward_dict, done, info
        """
        # 1. Convert Dictionary Actions to Global Action Vector
        global_action = self._map_actions(action_dict)
        
        # 2. Step the Environment
        global_obs, global_reward, done, info = self.raw_env.step(global_action)
        
        # 3. Construct Local Observations & Rewards
        obs_dict = self._extract_local_obs(global_obs)
        reward_dict = self._extract_local_rewards(global_obs, global_reward)
        
        return obs_dict, reward_dict, done, info

    def _extract_local_obs(self, global_obs):
        """
        Constructs local observation for each agent.
        Vector: [Local_Bus_Voltage, Internal_State]
        """
        obs_dict = {}
        
        # Helper to get voltage magnitude
        def get_v(bus):
            v_complex = global_obs['bus_voltages'][bus]
            # Average phase voltages
            return np.mean([v_complex[i] for i in range(len(v_complex)) if i%2==0])

        for agent_id in self.agents:
            atype = self.agent_types[agent_id]
            comp_name = self.agent_mapping[agent_id]
            
            local_obs = []
            
            if atype == 'cap':
                # State: [Voltage, Status(0/1)]
                comp = self.raw_env.circuit.capacitors[comp_name]
                v = get_v(comp.bus1)
                status = global_obs['cap_statuses'][comp_name]
                local_obs = [v, status]
                
            elif atype == 'reg':
                # State: [Voltage_Downstream, Tap_Position]
                comp = self.raw_env.circuit.regulators[comp_name]
                v = get_v(comp.bus2) # Regulate downstream
                tap = global_obs['reg_statuses'][comp_name]
                local_obs = [v, tap]
                
            elif atype == 'bat':
                # State: [Voltage, SOC] (Ignoring discharge rate for simplicity here)
                comp = self.raw_env.circuit.batteries[comp_name]
                v = get_v(comp.bus1)
                soc = global_obs['bat_statuses'][comp_name][0]
                local_obs = [v, soc]
                
            obs_dict[agent_id] = np.array(local_obs, dtype=np.float32)
            
        return obs_dict

    def _extract_local_rewards(self, global_obs, global_reward):
        """
        Custom Reward Logic:
        Local Reward = -1 * |Local_Voltage_Violation| + (Alpha * Global_Reward)
        
        This makes agents selfish (fix their own voltage) but cooperative (reduce system loss).
        """
        reward_dict = {}
        
        # Helper for violation
        def get_vio(bus):
            v_complex = global_obs['bus_voltages'][bus]
            v_mags = [v_complex[i] for i in range(len(v_complex)) if i%2==0]
            max_v = max(v_mags)
            min_v = min(v_mags)
            # Penalty if outside 0.95 - 1.05
            vio = max(0, max_v - 1.05) + max(0, 0.95 - min_v)
            return vio

        for agent_id in self.agents:
            atype = self.agent_types[agent_id]
            comp_name = self.agent_mapping[agent_id]
            
            target_bus = None
            if atype == 'cap': target_bus = self.raw_env.circuit.capacitors[comp_name].bus1
            elif atype == 'reg': target_bus = self.raw_env.circuit.regulators[comp_name].bus2
            elif atype == 'bat': target_bus = self.raw_env.circuit.batteries[comp_name].bus1
            
            local_violation = get_vio(target_bus)
            
            # Mix local selfish object with global objective
            # Heavy penalty for local violation, small bonus for global power loss reduction
            reward_dict[agent_id] = -(local_violation * 100.0) + (global_reward * 0.1)
            
        return reward_dict

    def _map_actions(self, action_dict):
        """
        Reconstructs the global action vector expected by PowerGym.
        Order: [Caps... , Regs..., Bats...]
        """
        # Initialize lists
        cap_actions = []
        reg_actions = []
        bat_actions = []
        
        # We must iterate in the ORDER defined in Env (env.cap_names, etc.)
        # to match the vector indices.
        
        for name in self.raw_env.cap_names:
            agent_id = f"cap_{name}"
            # Default to no-change or current state if missing
            act = action_dict.get(agent_id, 0) # Default 0 (Open)
            cap_actions.append(act)
            
        for name in self.raw_env.reg_names:
            agent_id = f"reg_{name}"
            # Default middle tap
            act = action_dict.get(agent_id, self.raw_env.reg_act_num // 2)
            reg_actions.append(act)
            
        for name in self.raw_env.bat_names:
            agent_id = f"bat_{name}"
            # Default idle
            if self.raw_env.bat_act_num == np.inf:
                act = action_dict.get(agent_id, 0.0)
            else:
                act = action_dict.get(agent_id, self.raw_env.bat_act_num // 2)
            bat_actions.append(act)
            
        # Concatenate
        return np.concatenate([cap_actions, reg_actions, bat_actions])
