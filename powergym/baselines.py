
import numpy as np

class LocalControlAgent:
    """
    A heuristic agent implementing local Volt-Var control logic.
    - Capacitors: Switch ON if local voltage < 0.95, OFF if > 1.05.
    - Regulators: Tap UP if downstream voltage < 0.95, DOWN if > 1.05.
    - Batteries: Idle (no discharge/charge).
    """
    def __init__(self, env):
        self.env = env
        self.cap_names = env.cap_names
        self.reg_names = env.reg_names
        self.bat_names = env.bat_names
        
    def act(self, obs=None):
        # Note: We access env.circuit directly to mimic "local sensing" 
        # without parsing the complex flattened observation vector.
        # In a real deployment, these would be local sensor readings.
        
        actions = []
        
        # 1. Capacitor Control
        for cap_name in self.cap_names:
            cap = self.env.circuit.capacitors[cap_name]
            # Check voltage at the capacitor's bus
            v_pu = self._get_avg_voltage(cap.bus1)
            
            curr_status = cap.status
            new_status = curr_status
            
            if v_pu < 0.95:
                new_status = 1 # ON
            elif v_pu > 1.05:
                new_status = 0 # OFF
            
            actions.append(int(new_status))
            
        # 2. Regulator Control
        for reg_name in self.reg_names:
            reg = self.env.circuit.regulators[reg_name]
            # Check voltage at the downstream bus (bus2)
            v_pu = self._get_avg_voltage(reg.bus2)
            
            # Current Tap Index
            # tap_feature: [mintap, maxtap, numtaps]
            mintap, maxtap, numtaps = reg.tap_feature
            step = (maxtap - mintap) / numtaps
            curr_tap_idx = int(round((reg.tap - mintap) / step))
            
            new_tap_idx = curr_tap_idx
            
            if v_pu < 0.95:
                new_tap_idx += 1 # Boost voltage
            elif v_pu > 1.05:
                new_tap_idx -= 1 # Lower voltage
                
            # Clamp to limits
            new_tap_idx = max(0, min(self.env.reg_act_num - 1, new_tap_idx))
            actions.append(int(new_tap_idx))
            
        # 3. Battery Control (Idle)
        if self.env.bat_act_num == np.inf:
            # Continuous: 0.0 means 0 kW
            actions.extend([0.0] * self.env.bat_num)
        else:
            # Discrete: Middle index means 0 kW
            idle_idx = self.env.bat_act_num // 2
            actions.extend([idle_idx] * self.env.bat_num)
            
        return actions

    def _get_avg_voltage(self, bus_name):
        # Helper to get average PU voltage magnitude across phases
        v_complex = self.env.circuit.bus_voltage(bus_name)
        # dss returns [mag1, ang1, mag2, ang2...]
        v_mags = [v_complex[i] for i in range(len(v_complex)) if i % 2 == 0]
        if not v_mags: return 1.0
        return sum(v_mags) / len(v_mags)


class IndependentMultiAgent:
    """
    A Decentralized Heuristic Agent.
    
    Acts solely on local observations provided by the MultiAgentPowerGrid environment.
    Does NOT access the global circuit state.
    
    Observation format per agent: [Local_Voltage, Internal_State]
    """
    def __init__(self, agent_types, reg_act_num, bat_act_num):
        self.agent_types = agent_types # Dict: agent_id -> 'cap', 'reg', 'bat'
        self.reg_act_num = reg_act_num
        self.bat_act_num = bat_act_num

    def act(self, obs_dict):
        """
        Args:
            obs_dict: {agent_id: np.array([voltage, state])}
        Returns:
            action_dict: {agent_id: action}
        """
        action_dict = {}
        
        for agent_id, obs in obs_dict.items():
            atype = self.agent_types[agent_id]
            v_pu = obs[0]
            current_state = obs[1]
            
            if atype == 'cap':
                # Cap Logic: V < 0.95 -> ON(1), V > 1.05 -> OFF(0)
                # If stable, maintain current state
                if v_pu < 0.95:
                    act = 1
                elif v_pu > 1.05:
                    act = 0
                else:
                    act = int(current_state)
                action_dict[agent_id] = act
                
            elif atype == 'reg':
                # Reg Logic: V < 0.95 -> TapUp, V > 1.05 -> TapDown
                # current_state is the Tap Index
                tap = int(current_state)
                
                if v_pu < 0.95:
                    tap += 1
                elif v_pu > 1.05:
                    tap -= 1
                    
                # Clamp
                tap = max(0, min(self.reg_act_num - 1, tap))
                action_dict[agent_id] = tap
                
            elif atype == 'bat':
                # Bat Logic: Idle
                if self.bat_act_num == np.inf:
                    action_dict[agent_id] = 0.0
                else:
                    action_dict[agent_id] = self.bat_act_num // 2
                    
        return action_dict
