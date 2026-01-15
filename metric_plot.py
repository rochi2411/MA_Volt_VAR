import matplotlib.pyplot as plt
import numpy as np

# --- Data Entry ---
systems = ['13-Bus', '34-Bus', '123-Bus']
x = np.arange(len(systems))
width = 0.25  # Width of the bars

# 1. Average Reward (Higher is better / Closer to 0)
reward_heuristic = [-35.12, -85.46, -356.47]
reward_monolithic = [-41.37, -80.60, -73.49]
reward_specialist = [-0.54, -54.62, -5.01]

# 2. Voltage Violation (Lower is better)
viol_heuristic = [0.1125, 0.1422, 0.1937]
viol_monolithic = [0.058, 0.1283, 0.1507]
viol_specialist = [0.0247, 0.0596, 0.0586]

# 3. Power Loss (Lower is better)
loss_heuristic = [0.0187, 0.0684, 0.0149]
loss_monolithic = [0.0177, 0.071, 0.0158]
loss_specialist = [0.0188, 0.084, 0.016]

# --- Helper Function to Make Plots ---
def create_bar_plot(data1, data2, data3, title, ylabel, filename, better_direction="lower"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    rects1 = ax.bar(x - width, data1, width, label='Heuristic', color='gray', alpha=0.6)
    rects2 = ax.bar(x, data2, width, label='Monolithic', color='blue', alpha=0.7)
    rects3 = ax.bar(x + width, data3, width, label='Multi-Agent (Proposed)', color='orange', alpha=0.9)

    # Labels and Title
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate bars with values
    # We adjust label position based on whether values are positive or negative
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            xy_pos = (rect.get_x() + rect.get_width() / 2, height)
            
            # Text offset: if bar is negative, put text below it; if positive, above it
            if height < 0:
                xy_text = (0, -12) 
            else:
                xy_text = (0, 3)

            ax.annotate(f'{height:.2f}',
                        xy=xy_pos,
                        xytext=xy_text,
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# --- Generate the 3 Plots ---

# Plot 1: Average Reward
create_bar_plot(reward_heuristic, reward_monolithic, reward_specialist, 
                "Average Episode Reward", 
                "Reward Value", 
                "Compare_Reward.png")

# Plot 2: Voltage Violation
create_bar_plot(viol_heuristic, viol_monolithic, viol_specialist, 
                "Voltage Violation Index", 
                "Violation Magnitude", 
                "Compare_Violation.png")

# Plot 3: Power Loss
create_bar_plot(loss_heuristic, loss_monolithic, loss_specialist, 
                "Average Power Loss", 
                "Power Loss", 
                "Compare_Loss.png")
