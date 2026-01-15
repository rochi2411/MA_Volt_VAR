import gymnasium as gym
from gymnasium import spaces
import numpy as np
from powergym.ma_env import MultiAgentPowerGrid
import argparse
import os

# Note: This requires stable-baselines3
# pip install stable-baselines3 shimmy

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    import torch
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("Error: stable-baselines3 or torch not installed. Please install them to train agents.")
    exit(1)

# Helper function (formerly _get_agent_obs method)
def _get_agent_obs_helper(wrapper_instance, agent_id):
    raw_obs_volts_state = wrapper_instance.current_obs_dict[agent_id] # This is [Voltage, State] from ma_env
    
    # Normalize Voltage and State components
    normalized_raw = wrapper_instance._normalize_individual_obs(agent_id, raw_obs_volts_state)
    
    # Add Agent Type ID. Normalize type_id as well to [-1, 1] for Box space.
    atype = wrapper_instance.ma_env.agent_types[agent_id]
    if atype == 'cap': type_id = -1.0 # Cap: maps to -1
    elif atype == 'reg': type_id = 0.0 # Reg: maps to 0
    elif atype == 'bat': type_id = 1.0 # Bat: maps to 1
    
    # Combine normalized obs and type_id.
    return np.array([normalized_raw[0], normalized_raw[1], type_id], dtype=np.float32)


class IPPO_Wrapper(gym.Env):
    # Added a test comment to force re-parsing of the class definition
    """
    Wraps the Multi-Agent environment to make it look like a single-agent environment
    for Independent PPO (IPPO) training.
    
    In IPPO, all agents share the same policy network (parameter sharing).
    We treat every agent's step as a sample for the PPO trainer.
    """
    def __init__(self, env_name='13Bus'):
        super().__init__() # Call parent constructor for gym.Env
        self.ma_env = MultiAgentPowerGrid(env_name)
        self.agents = self.ma_env.agents
        
        # Manually define original observation space bounds for normalization
        # based on agent type.
        self.original_ma_obs_bounds = {}
        for agent_id in self.agents:
            atype = self.ma_env.agent_types[agent_id]
            # Local observation is [Voltage, State]
            voltage_low = 0.8 # Typical lower bound for voltage pu
            voltage_high = 1.2 # Typical upper bound for voltage pu

            if atype == 'cap':
                # Capacitor state is 0 or 1
                state_low = 0.0
                state_high = 1.0
            elif atype == 'reg':
                # Regulator state is tap position from 0 to reg_act_num-1
                state_low = 0.0
                state_high = float(self.ma_env.raw_env.reg_act_num - 1)
            elif atype == 'bat':
                # Battery state is SOC from 0 to 1
                state_low = 0.0
                state_high = 1.0
            else:
                raise ValueError(f"Unknown agent type: {atype}")
            
            low = np.array([voltage_low, state_low], dtype=np.float32)
            high = np.array([voltage_high, state_high], dtype=np.float32)
            
            self.original_ma_obs_bounds[agent_id] = {
                'low': low,
                'high': high,
                'range': high - low
            }
        
        self.original_ma_act_spaces = self.ma_env.action_spaces # Dictionary of agent_id -> Discrete space

        # Unified observation space: [Normalized Voltage, Normalized State, AgentTypeID]
        # AgentTypeID: -1=Cap, 0=Reg, 1=Bat (normalized to [-1,1] range by these values)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Unified action space: Discrete(33) (Max of Reg/Bat). Caps will use 0/1.
        self.action_space = gym.spaces.Discrete(33) 
        
        self.current_obs_dict = None
        self.agent_idx = 0
        
    def _normalize_individual_obs(self, agent_id, obs):
        # Normalize obs (Voltage, State) to [-1, 1] range
        bounds = self.original_ma_obs_bounds[agent_id]
        low = bounds['low']
        high = bounds['high']
        obs_range = bounds['range'] + 1e-8 # Add epsilon to prevent division by zero
        
        normalized_obs = (obs - low) / obs_range * 2 - 1
        return normalized_obs.astype(np.float32)

    def reset(self, *, seed=None, options=None): # Gymnasium API
        if seed is not None:
            self.ma_env.raw_env.seed(seed) # Set seed for the underlying powergym Env
        self.current_obs_dict = self.ma_env.reset()
        self.agent_idx = 0
        return _get_agent_obs_helper(self, self.agents[0]), {} # Return observation and empty info dict

    def step(self, action):
        current_agent = self.agents[self.agent_idx]
        
        if not hasattr(self, 'action_buffer'):
            self.action_buffer = {}
            
        self.action_buffer[current_agent] = action
        
        self.agent_idx += 1
        
        if self.agent_idx < len(self.agents):
            next_agent = self.agents[self.agent_idx]
            obs = _get_agent_obs_helper(self, next_agent)
            return obs, 0.0, False, False, {} # Gymnasium API: obs, reward, terminated, truncated, info
            
        else:
            # ALL agents have acted. Step the real environment.
            # Map the Discrete(33) actions back to agent-specific actions
            mapped_actions = {}
            for agent_id, raw_action_from_ppo in self.action_buffer.items():
                atype = self.ma_env.agent_types[agent_id]
                if atype == 'cap':
                    # Capacitors have Discrete(2) action space (0 or 1)
                    mapped_actions[agent_id] = raw_action_from_ppo % 2
                elif atype == 'reg':
                    # Regulators have Discrete(reg_act_num)
                    # Ensure action is within bounds
                    max_reg_action = self.ma_env.raw_env.reg_act_num - 1
                    mapped_actions[agent_id] = min(raw_action_from_ppo, max_reg_action)
                elif atype == 'bat':
                    # Batteries have Discrete(bat_act_num) or Box
                    if self.ma_env.raw_env.bat_act_num == np.inf: # Continuous
                        # PPO output for Discrete(33) cannot directly map to continuous [-1,1]
                        # For now, let's treat the discrete action as a scaled continuous action
                        # Map 0-32 to -1.0 to 1.0 (approx)
                        mapped_actions[agent_id] = (raw_action_from_ppo / 32.0) * 2 - 1
                        # This needs to be an array for Box action space
                        mapped_actions[agent_id] = np.array([mapped_actions[agent_id]], dtype=np.float32)
                    else: # Discrete
                        max_bat_action = self.ma_env.raw_env.bat_act_num - 1
                        mapped_actions[agent_id] = min(raw_action_from_ppo, max_bat_action)
                else:
                    raise ValueError(f"Unknown agent type for action mapping: {atype}")
            
            obs_dict, reward_dict, done, info = self.ma_env.step(mapped_actions)
            
            self.action_buffer = {}
            self.agent_idx = 0
            self.current_obs_dict = obs_dict
            
            avg_reward = np.mean(list(reward_dict.values()))
            
            next_agent = self.agents[0]
            obs = _get_agent_obs_helper(self, next_agent)
            
            return obs, avg_reward, done, False, info # Gymnasium API: obs, reward, terminated, truncated, info

def train_ippo(env_name='13Bus', steps=10000):
    print(f"Training IPPO (Specialist Ensemble) on {env_name}...")
    
    env = IPPO_Wrapper(env_name)
    # Wrap with Monitor for logging
    log_dir = f"logs/specialist_ensemble_{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir + "monitor.csv") # Apply monitor to the IPPO_Wrapper
    
    model = PPO("MlpPolicy", env, verbose=1)
    
    model.learn(total_timesteps=steps)
    
    model.save(f"specialist_ensemble_{env_name}")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--env_name', default='13Bus')
    parser.add_argument('--steps', type=int, default=10000, help='Total training timesteps')
    args = parser.parse_args()
    train_ippo(args.env_name, args.steps)