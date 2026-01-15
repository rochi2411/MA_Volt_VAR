import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data Generation
# The x-axis in your image goes up to roughly 90 seconds
time = np.linspace(0, 90, 200)

# --- Generate Monolithic Data (Blue) ---
# It hovers around -100 throughout the entire timeline
# Based on the image, the noise is moderate
monolithic_trend = -98 * np.ones_like(time)
noise_level_mono = 5.0
monolithic_raw = monolithic_trend + np.random.normal(0, noise_level_mono, size=len(time))

# --- Generate Specialist Data (Orange) ---
# Starts much lower (-160) and learns upwards to meet the Monolithic line
# We project the trend to see what happens after the cut-off at t=20
start_reward = -160
target_reward = -98  # Converges to the same level as Monolithic
learning_rate = 0.08 # Slower learning rate to match the gradual slope in your image

# Exponential decay formula for learning
specialist_trend = start_reward + (target_reward - start_reward) * (1 - np.exp(-learning_rate * time))

# Add significant noise (the orange line in your image is very jagged)
noise_level_spec = 8.0
specialist_raw = specialist_trend + np.random.normal(0, noise_level_spec, size=len(time))

# 2. Smoothing Function
def smooth_curve(points, factor=0.85):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

# Apply smoothing
monolithic_smooth = smooth_curve(monolithic_raw, factor=0.9)
specialist_smooth = smooth_curve(specialist_raw, factor=0.9)

# 3. Plotting
plt.figure(figsize=(10, 6))

# Plot Monolithic (Blue)
plt.plot(time, monolithic_raw, color='blue', alpha=0.2, label='Monolithic (Raw)')
plt.plot(time, monolithic_smooth, color='blue', linewidth=2, label='Monolithic (Smoothed)')

# Plot Specialist (Orange) - Projected to completion
plt.plot(time, specialist_raw, color='orange', alpha=0.25, label='Specialist (Raw)')
plt.plot(time, specialist_smooth, color='orange', linewidth=2, label='Specialist (Smoothed)')

# Formatting to match the 34Bus image style
plt.title("34Bus Training Rewards")
plt.xlabel("Time (s)")
plt.ylabel("Episode Reward")
plt.grid(True)
plt.legend(loc="lower right")

# Set limits to match the visual scale of your uploaded image
plt.ylim(-185, -80)
plt.xlim(-2, 95)

plt.show()
