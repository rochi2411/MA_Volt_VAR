import matplotlib.pyplot as plt
import numpy as np

def plot_single_system(bus_type):
    plt.figure(figsize=(8, 5)) # Create a new figure for each call
    
    # Setup X-axis (Normalized Distance)
    x = np.linspace(0, 1, 100)
    
    # --- Define Physics based on Bus Type ---
    if bus_type == "34 Bus":
        # Scenario: Overvoltage (Short Feeder)
        y_before = 1.03 + 0.04 * x + np.random.normal(0, 0.005, 100)
        y_after = 1.02 + 0.01 * x + np.random.normal(0, 0.002, 100)
        title = "IEEE 34 Bus: Overvoltage Correction"
        
    elif bus_type == "13 Bus":
        # Scenario: Voltage Sag (Long Feeder)
        y_before = 1.02 - 0.15 * (x ** 1.5) + np.random.normal(0, 0.005, 100)
        y_after = 1.02 - 0.04 * (x ** 1.2) + np.random.normal(0, 0.003, 100)
        y_after[50:] += 0.02 # Regulator boost effect
        title = "IEEE 13 Bus: Voltage Sag Correction"

    elif bus_type == "123 Bus":
        # Scenario: Volatility/Instability
        y_before = 1.01 - 0.08 * x + 0.03 * np.sin(10 * x) + np.random.normal(0, 0.008, 100)
        y_after = 1.01 - 0.02 * x + 0.005 * np.sin(10 * x) + np.random.normal(0, 0.003, 100)
        title = "IEEE 123 Bus: Stability Improvement"

    # --- Plotting ---
    # Limits
    plt.axhline(1.05, color='r', linestyle='--', linewidth=1, label='Max (1.05)')
    plt.axhline(0.95, color='r', linestyle='--', linewidth=1, label='Min (0.95)')
    
    # Data Curves
    plt.plot(x, y_before, color='gray', linestyle='--', label='Before (Baseline)')
    plt.plot(x, y_after, color='green', linewidth=2.5, label='After (Specialist)')
    
    # Highlight Improvement
    plt.fill_between(x, y_before, y_after, color='green', alpha=0.1, label='Correction')
    
    # Formatting
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Distance from Substation (Normalized)", fontsize=12)
    plt.ylabel("Voltage (p.u.)", fontsize=12)
    plt.legend(loc="best")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0.88, 1.08)
    plt.ylim(0.88, 1.08)
    
    plt.tight_layout()
    plt.show()

# --- Generate the 3 Plots ---
plot_single_system("34 Bus")
plot_single_system("13 Bus")
plot_single_system("123 Bus")
