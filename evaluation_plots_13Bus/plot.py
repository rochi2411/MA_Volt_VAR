import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data Generation
# We simulate 45 seconds of training time (matching the x-axis of your image)
time = np.linspace(0, 45, 150)  # 150 data points simulating episodes

# --- Generate Monolithic Data (Blue) ---
# Stable performance around -43 with some noise
noise_level_mono = 3.0
monolithic_trend = -43 * np.ones_like(time)
# Add random noise for the "Raw" look
monolithic_raw = monolithic_trend + np.random.normal(0, noise_level_mono, size=len(time))

# --- Generate Specialist Data (Orange) ---
# Starts at -100, improves (learns) over time, and converges towards -40
# We use an exponential decay function to mimic the "learning curve"
start_reward = -100
target_reward = -42
learning_rate = 0.15  # Controls how fast it learns

# Formula: Start + (Target - Start) * (1 - e^(-rate * time))
specialist_trend = start_reward + (target_reward - start_reward) * (1 - np.exp(-learning_rate * time))

# Add higher noise initially (common in RL exploration) that decreases slightly
noise_level_spec = 5.0 
specialist_raw = specialist_trend + np.random.normal(0, noise_level_spec, size=len(time))

# 2. Smoothing Function (Exponential Moving Average)
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

# Apply smoothing
monolithic_smooth = smooth_curve(monolithic_raw, factor=0.85)
specialist_smooth = smooth_curve(specialist_raw, factor=0.85)

# 3. Plotting
plt.figure(figsize=(10, 6))

# Plot Monolithic (Blue)
plt.plot(time, monolithic_raw, color='blue', alpha=0.2, label='Monolithic (Raw)')
plt.plot(time, monolithic_smooth, color='blue', linewidth=2, label='Monolithic (Smoothed)')

# Plot Specialist (Orange) - Now Complete!
plt.plot(time, specialist_raw, color='orange', alpha=0.25, label='Specialist (Raw)')
plt.plot(time, specialist_smooth, color='orange', linewidth=2, label='Specialist (Smoothed)')

# Formatting to match your original image
plt.title("13Bus Training Rewards")
plt.xlabel("Time (s)")
plt.ylabel("Episode Reward")
plt.grid(True)
plt.legend(loc="lower right")

# Set limits to match your image
plt.ylim(-105, -28)
plt.xlim(-0.5, 45)

plt.show()
