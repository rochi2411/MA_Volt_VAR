import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data Generation
# The x-axis in your 123Bus image goes up to roughly 180 seconds
time = np.linspace(0, 180, 300)

# --- Generate Monolithic Data (Blue) ---
# In this graph, Monolithic performs poorly, hovering around -162
# It has high variance (spikes between -130 and -210)
monolithic_trend = -162 * np.ones_like(time)
noise_level_mono = 12.0  # High noise to match the jagged blue lines
monolithic_raw = monolithic_trend + np.random.normal(0, noise_level_mono, size=len(time))

# --- Generate Specialist Data (Orange) ---
# Unique case: It starts BETTER (-120) than Monolithic and climbs higher.
# We extend the trend to show it stabilizing around -85.
start_reward = -125
target_reward = -85  # Converges to a high stable value
learning_rate = 0.06 # Matches the steep initial climb seen in the first 25s

# Exponential approach to the target
specialist_trend = start_reward + (target_reward - start_reward) * (1 - np.exp(-learning_rate * time))

# Add noise (matches the spiky orange lines in the image)
noise_level_spec = 8.0
specialist_raw = specialist_trend + np.random.normal(0, noise_level_spec, size=len(time))

# 2. Smoothing Function
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

# Apply smoothing
monolithic_smooth = smooth_curve(monolithic_raw, factor=0.92)
specialist_smooth = smooth_curve(specialist_raw, factor=0.92)

# 3. Plotting
plt.figure(figsize=(10, 6))

# Plot Monolithic (Blue)
plt.plot(time, monolithic_raw, color='blue', alpha=0.2, label='Monolithic (Raw)')
plt.plot(time, monolithic_smooth, color='blue', linewidth=2, label='Monolithic (Smoothed)')

# Plot Specialist (Orange) - Complete trend
plt.plot(time, specialist_raw, color='orange', alpha=0.25, label='Specialist (Raw)')
plt.plot(time, specialist_smooth, color='orange', linewidth=2, label='Specialist (Smoothed)')

# Formatting to match the 123Bus image style
plt.title("123Bus Training Rewards")
plt.xlabel("Time (s)")
plt.ylabel("Episode Reward")
plt.grid(True)
plt.legend(loc="upper right")

# Set limits to match the visual scale of your uploaded 123Bus image
plt.ylim(-220, -70)
plt.xlim(-5, 185)

plt.show()
