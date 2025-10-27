# ---------------------------------------------
# Import Required Libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.animation import FuncAnimation
warnings.filterwarnings("ignore")

# Set a seed for reproducibility
# np.random.seed(42)

# ---------------------------------------------
# Define the Custom Objective Function
# ---------------------------------------------
def objective_function(x, start, end, lambda_bend, warp_field):
    """
    Objective function simulating space-time bending effort to reach a goal.
    Includes:
      - Geodesic-like distance (with warp distortion)
      - Bending penalty (simulated curvature resistance)
      - Time dilation / traversal cost (effort adjusted by warp)
      - Warp field energy cost
    """
    # Distance to target with warp influence
    distance = np.linalg.norm((x - end) * (1 + warp_field))

    # Bending effort (quadratic penalty)
    bending_cost = lambda_bend * np.sum((x - start)**2)

    # Traversal cost (like time dilation)
    effort_cost = np.sum(np.abs(x - start) * (1 + warp_field))

    # Warp field maintenance energy
    energy_cost = np.sum(warp_field**2)

    # Total cost
    return distance + bending_cost + 0.5 * effort_cost + 0.2 * energy_cost

# ---------------------------------------------
# Firefly Algorithm Parameters
# ---------------------------------------------
num_fireflies = 100
num_dimensions = 2
num_iterations = 50

start = np.array([0, 0])
end = np.array([10, 10])
lambda_bend = 0.1
warp_field = np.random.uniform(0.1, 0.5, size=num_dimensions)  # Static warp field

alpha = 0.3   # Randomness strength
beta0 = 1.0   # Base attractiveness
gamma = 0.5   # Absorption coefficient

# ---------------------------------------------
# Initialize Fireflies
# ---------------------------------------------
positions = np.random.uniform(-5, 15, size=(num_fireflies, num_dimensions))
intensities = np.array([objective_function(p, start, end, lambda_bend, warp_field) for p in positions])

# Record best fitness for plotting
best_costs = []

# Store positions for animation
positions_history = []

# ---------------------------------------------
# Firefly Optimization Loop
# ---------------------------------------------
print("Starting Firefly Optimization...\n")

for iteration in range(num_iterations):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if intensities[j] < intensities[i]:
                r = np.linalg.norm(positions[i] - positions[j])
                beta = beta0 * np.exp(-gamma * r**2)

                # Move firefly i towards firefly j
                step = beta * (positions[j] - positions[i]) + alpha * (np.random.rand(num_dimensions) - 0.5)
                positions[i] += step

                # Optional: Keep positions bounded (for educational clarity)
                positions[i] = np.clip(positions[i], -10, 20)

        # Recalculate intensity after movement
        intensities[i] = objective_function(positions[i], start, end, lambda_bend, warp_field)

    # Track best cost
    best_idx = np.argmin(intensities)
    best_costs.append(intensities[best_idx])
    
    # Save positions for animation
    positions_history.append(positions.copy())

    # Progress print
    print(f"Iteration {iteration + 1:02d}/{num_iterations} | Best Fitness: {intensities[best_idx]:.4f}")

# ---------------------------------------------
# Final Optimization Results
# ---------------------------------------------
best_position = positions[best_idx]
print("\nOptimization Completed!")
print(f"Best firefly index: {best_idx}") # the best firefly
print(f"Global Best Position: {best_position}")# position of the best firefly
print(f"Final Objective Value: {intensities[best_idx]:.4f}") # minimum total energy cost (to find the best path)

# ---------------------------------------------
# Plot Cost Convergence Over Iterations
# ---------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(best_costs, linewidth=2, marker='o', color='black')
plt.title(" Convergence of Firefly Algorithm with Space-Time Analogy")
plt.xlabel("Iteration")
plt.ylabel("Best Cost")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Final Firefly Positions Visualization
# ---------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], label="Final Fireflies", color="blue", s=60)
plt.scatter(best_position[0], best_position[1], label="Global Best", color="red", marker="x", s=120)
plt.scatter(end[0], end[1], label="Target Position", color="green", marker="*", s=200)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(" Final Firefly Distribution in 2D Search Space")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Animate Firefly Movements Over Iterations
# ---------------------------------------------
print("\nCreating animation of firefly movement...")

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter([], [], s=60, color='blue', label='Fireflies')
best_sc = ax.scatter([], [], s=120, color='red', marker='x', label='Global Best')
target_sc = ax.scatter(end[0], end[1], s=200, color='green', marker='*', label='Target Position')

ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Firefly Movement Animation")
ax.grid(True)
ax.legend()

def update(frame):
    positions = positions_history[frame]
    sc.set_offsets(positions)
    best_idx = np.argmin([objective_function(p, start, end, lambda_bend, warp_field) for p in positions])
    best_sc.set_offsets(positions[best_idx])
    return sc, best_sc

ani = FuncAnimation(fig, update, frames=len(positions_history), interval=100, repeat=False)

plt.tight_layout()
plt.show()

# To save animation:
# ani.save("firefly_animation.mp4", writer="ffmpeg", fps=5)
# ani.save("firefly_animation.gif", writer="pillow", fps=5)
