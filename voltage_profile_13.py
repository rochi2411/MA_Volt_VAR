import numpy as np
import matplotlib.pyplot as plt

def generate_34bus_profile():
    # Distance normalized (0 = Substation, 1 = End of Line)
    # The IEEE 34 bus is very long, so we simulate distance steps
    dist = np.linspace(0, 1, 500)
    
    # --- 1. Baseline (Grey): The "Sag" ---
    # Without control, voltage drops continuously due to impedance
    # Starts at 1.0, drops deep to ~0.88 (Violation)
    v_base = 1.0 - (0.12 * dist) - (0.05 * dist**2)
    # Add noise
    v_base += np.random.normal(0, 0.002, 500)
    
    # --- 2. Corrected (Green): The "Sawtooth" ---
    # Regulators are located at Substation, ~30% down line, and ~60% down line
    v_corr = np.zeros_like(dist)
    
    current_v = 1.03 # Start slightly boosted
    
    for i, d in enumerate(dist):
        # Natural line drop
        drop = 0.0006 + (0.0001 * d) # Drop per step
        current_v -= drop
        
        # Regulator 1 Action (at d=0.35)
        # Boost back up
        if 0.34 < d < 0.35:
            current_v += 0.04
            
        # Regulator 2 Action (at d=0.75)
        # Boost again to save the tail end
        if 0.74 < d < 0.75:
            current_v += 0.035
            
        # Add noise
        v_corr[i] = current_v + np.random.normal(0, 0.001)

    # ================= PLOTTING =================
    plt.figure(figsize=(8, 5))
    
    # Safety Limits
    plt.axhline(1.05, color='red', linestyle='--', linewidth=1, label='Max (1.05)')
    plt.axhline(0.95, color='red', linestyle='--', linewidth=1, label='Min (0.95)')
    
    # Profiles
    plt.plot(dist, v_base, color='gray', linestyle='--', linewidth=2, label='Baseline (No Control)')
    plt.plot(dist, v_corr, color='green', linewidth=2.5, label='Multi-Agent Control')
    
    # Highlight the Boosts
    plt.fill_between(dist, v_base, v_corr, color='green', alpha=0.1, label='Voltage Correction')
    
    # Annotations for Regulators
    plt.annotate('Inline Reg 1\nBoost', xy=(0.35, 1.02), xytext=(0.25, 1.06),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)
    plt.annotate('Inline Reg 2\nBoost', xy=(0.75, 0.99), xytext=(0.65, 1.03),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)

    plt.title("IEEE 34-Bus: Long Feeder Voltage Correction", fontsize=14, fontweight='bold')
    plt.ylabel("Voltage (p.u.)", fontsize=12)
    plt.xlabel("Distance from Substation (Normalized)", fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Zoom y-axis to show detail
    plt.ylim(0.85, 1.10)
    
    plt.tight_layout()
    plt.savefig('Voltage_profile_34.pdf', dpi=300) # Saving as PDF for your LaTeX
    print("Plot generated: 'Voltage_profile_34.pdf'")
    plt.show()

if __name__ == "__main__":
    generate_34bus_profile()
