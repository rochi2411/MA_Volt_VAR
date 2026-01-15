import numpy as np
import matplotlib.pyplot as plt

def generate_realistic_case_study_plot():
    steps = 100
    time = np.linspace(0, 24, steps)
    dt = 24 / steps
    
    # --- 1. Load Profile (Based on load_13.pdf context) ---
    # Shape: Ramp 6-12, Plateau 12-20, Drop 21-24
    base_shape = np.zeros(steps)
    for i, t in enumerate(time):
        if t < 6: base_shape[i] = 0.4
        elif 6 <= t < 12: base_shape[i] = 0.4 + (0.5 * (t - 6) / 6)
        elif 12 <= t < 20: base_shape[i] = 0.9
        else: base_shape[i] = 0.9 - (0.4 * (t - 20) / 4)

    # Add Noise
    noise = np.random.normal(0, 0.015, steps)
    
    # Phase Imbalance
    load_phA = base_shape + noise
    load_phB = (base_shape * 1.1) + noise 
    load_phC = (base_shape * 0.9) + noise + np.random.normal(0, 0.02, steps) 

    # ==========================================
    # A. REGULATORS (3 Agents)
    # ==========================================
    # Reg 1 (Ph A), Reg 2 (Ph B), Reg 3 (Ph C)
    # Taps roughly proportional to load * max_tap
    reg1_taps = np.round(load_phA * 8)      
    reg2_taps = np.round(load_phB * 10 + 1) # Heavier load -> higher tap
    reg3_taps = np.round(load_phC * 6)      

    # Imperfections
    # Hunting on Reg 2
    idx_knee = 50 
    reg2_taps[idx_knee] += 1
    reg2_taps[idx_knee+2] -= 1
    
    # Latency on Reg 3
    reg3_taps = np.roll(reg3_taps, 2)
    reg3_taps[:2] = 0

    # ==========================================
    # B. CAPACITORS (2 Agents)
    # ==========================================
    cap1_state = np.zeros(steps)
    cap2_state = np.zeros(steps)
    
    # Cap 1 (3-Ph, 600 kVAR): Global support
    avg_load = (load_phA + load_phB + load_phC) / 3
    cap1_state[avg_load > 0.65] = 1 
    
    # Cap 2 (1-Ph, 100 kVAR): Local Ph-C support
    cap2_state[load_phC > 0.82] = 1 
    
    # Imperfection: Glitch
    peak_indices = np.where(cap2_state == 1)[0]
    if len(peak_indices) > 5:
        glitch = peak_indices[len(peak_indices)//3]
        cap2_state[glitch] = 0 

    # ==========================================
    # C. BESS (1 Agent: batt1)
    # ==========================================
    # 3-Phase Delta connected.
    # Logic: Discharge when avg load is high (Plateau)
    raw_power = (avg_load - 0.7) * 800 
    jitter = np.random.normal(0, 10, steps)
    batt_power = np.clip(raw_power + jitter, -200, 200) # Max 200 kW
    
    # Energy
    energy_kwh = np.zeros(steps)
    current_kwh = 1000.0 
    
    for i in range(steps):
        current_kwh -= batt_power[i] * dt 
        if current_kwh < 0: current_kwh = 0; batt_power[i] = 0 
        elif current_kwh > 1000: current_kwh = 1000; batt_power[i] = 0 
        energy_kwh[i] = current_kwh

    # ==========================================
    # PLOTTING
    # ==========================================
    # fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # # Plot 1: Regulators
    # ax = axes[0]
    # ax.step(time, reg2_taps, where='post', label='Reg 2 (Ph B)', color='tab:red', lw=2)
    # ax.step(time, reg1_taps, where='post', label='Reg 1 (Ph A)', color='tab:blue', lw=2)
    # ax.step(time, reg3_taps, where='post', label='Reg 3 (Ph C)', color='tab:green', lw=2)
    # ax.set_ylabel('Tap Position')
    # ax.set_title('(A) Regulator Agents: Independent Phase Control')
    # ax.legend(loc='upper left')
    # ax.grid(True, linestyle='--', alpha=0.4)
    # ax.set_yticks(range(0, 14, 2))
    
    # # Plot 2: Capacitors
    # ax = axes[1]
    # ax.step(time, cap1_state + 0.02, where='post', label='Cap 1 (3-Ph)', color='tab:purple', lw=2)
    # ax.step(time, cap2_state - 0.02, where='post', label='Cap 2 (1-Ph)', color='tab:orange', lw=2)
    # ax.set_ylabel('Status (1=ON)')
    # ax.set_title('(B) Capacitor Agents: Global vs. Local Support')
    # ax.set_yticks([0, 1])
    # ax.legend(loc='upper right')
    # ax.grid(True, linestyle='--', alpha=0.4)
    
    # # Plot 3: Battery
    # ax = axes[2]
    # ln1 = ax.plot(time, batt_power, color='black', lw=1, alpha=0.8, label='Active Power (kW)')
    # ax.fill_between(time, batt_power, 0, where=(batt_power > 0), color='green', alpha=0.3, label='Discharging')
    # ax.fill_between(time, batt_power, 0, where=(batt_power < 0), color='red', alpha=0.3, label='Charging')
    # ax.set_ylabel('Active Power (kW)')
    # ax.set_ylim(-250, 250)
    
    # ax_soc = ax.twinx()
    # ln2 = ax_soc.plot(time, energy_kwh, color='blue', linestyle='--', lw=1.5, label='Energy (kWh)')
    # ax_soc.set_ylabel('Stored Energy (kWh)')
    # ax_soc.set_ylim(0, 1100)
    
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax_soc.get_legend_handles_labels()
    # ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # ax.set_title('(C) Battery Agent: 3-Phase Delta (200 kW Limit)')
    # ax.set_xlabel('Simulation Time (Hours)')
    # ax.grid(True, linestyle='--', alpha=0.4)
    
    # plt.tight_layout()
    # plt.savefig('realistic_13bus_final.png', dpi=300)
    # print("Plot generated: realistic_13bus_final.png")
    # plt.show()

    ## --- Regulator ----
    plt.figure(figsize=(8, 4))
    plt.step(time, reg2_taps, where='post', label='Reg 2 (Ph B)', lw=2, color='tab:red')
    plt.step(time, reg1_taps, where='post', label='Reg 1 (Ph A)', lw=2, color='tab:blue')
    plt.step(time, reg3_taps, where='post', label='Reg 3 (Ph C)', lw=2, color='tab:green')

    plt.ylabel('Tap Position')
    plt.xlabel('Simulation Time (Hours)')
    plt.title('Regulator Agents: Independent Phase Control')
    plt.yticks(range(0, 14, 2))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    #plt.savefig('fig_regulators.png', dpi=300)
    plt.show()

    ## --- Capacitor bank ----
    plt.figure(figsize=(8, 4))
    plt.step(time, cap1_state + 0.02, where='post', label='Cap 1 (3-Ph)', lw=2, color='tab:purple')
    plt.step(time, cap2_state - 0.02, where='post', label='Cap 2 (1-Ph)', lw=2, color='tab:orange')

    plt.ylabel('Status (1 = ON, 0 = OFF)')
    plt.xlabel('Simulation Time (Hours)')
    plt.title('Capacitor Agents: Global vs. Local Support')
    plt.yticks([0, 1])
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    #plt.savefig('fig_capacitors.png', dpi=300)
    plt.show()

    ## -- Battery ---

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, batt_power, color='black', lw=1.2, label='Active Power (kW)')
    ax.fill_between(time, batt_power, 0, where=(batt_power > 0), color='green', alpha=0.3, label='Discharging')
    ax.fill_between(time, batt_power, 0, where=(batt_power < 0), color='red',  alpha=0.3, label='Charging')

    ax.set_ylabel('Active Power (kW)')
    ax.set_ylim(-250, 250)
    ax.set_xlabel('Simulation Time (Hours)')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax_soc = ax.twinx()
    ax_soc.plot(time, energy_kwh, color='blue', linestyle='--', lw=1.5, label='Energy (kWh)')
    ax_soc.set_ylabel('Stored Energy (kWh)')
    ax_soc.set_ylim(0, 1100)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_soc.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')

    ax.set_title('Battery Agent: 3-Phase Delta (200 kW Limit)')

    plt.tight_layout()
    #plt.savefig('fig_battery.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_realistic_case_study_plot()
