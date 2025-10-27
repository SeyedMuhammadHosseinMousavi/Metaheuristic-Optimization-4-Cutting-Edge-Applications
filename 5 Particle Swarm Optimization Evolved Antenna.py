# ---------------------------------------------
# Import Required Libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
# np.random.seed(42)

# ---------------------------------------------
# Objective Function for Antenna Design
# ---------------------------------------------
def objective_function(antenna_points):
    """
    Evaluates the cost of a 3D antenna shape based on:
    - Total length of the antenna.
    - Bending smoothness (penalty for sharp z-changes).

    Parameters:
        antenna_points (ndarray): (joints+1, 3) coordinates of antenna joints.

    Returns:
        float: Total cost (lower is better).
    """
    # Total antenna length (Euclidean sum between segments)
    total_length = np.sum(np.linalg.norm(np.diff(antenna_points, axis=0), axis=1))

    # Smoothness penalty: large z-jumps = high penalty
    z_bend_penalty = np.sum(np.abs(np.diff(antenna_points[:, 2])))

    # Weighted cost function
    cost = total_length + 0.3 * z_bend_penalty
    return cost

# ---------------------------------------------
# Generate Initial Antenna (Optional)
# ---------------------------------------------
def generate_initial_antenna(joints=7):
    """
    Generate a random antenna with given joints progressing upward in Z.

    Returns:
        ndarray: (joints+1, 3) array of XYZ coordinates.
    """
    z = np.linspace(0, 10, joints + 1)
    x = np.random.uniform(-1, 1, joints + 1)
    y = np.random.uniform(-1, 1, joints + 1)
    return np.column_stack((x, y, z))

# ---------------------------------------------
# PSO Parameters
# ---------------------------------------------
num_particles = 10
num_iterations = 300
joints = 7
dimensions = (joints + 1) * 3  # Each antenna = (joints + 1) × 3D

# ---------------------------------------------
# Run PSO Function
# ---------------------------------------------
def run_pso():
    """
    Run Particle Swarm Optimization to find optimal antenna geometry.

    Returns:
        best_antenna (ndarray): Optimized antenna (shape: [joints+1, 3])
        cost_history (list): Global best cost per iteration
    """
    # Initialize particles (random antenna geometries)
    particles = np.random.uniform(-1, 1, (num_particles, dimensions))
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, dimensions))

    best_particle_positions = particles.copy()
    best_particle_costs = np.array([
        objective_function(p.reshape(-1, 3)) for p in particles
    ])
    global_best_position = particles[np.argmin(best_particle_costs)]
    global_best_cost = np.min(best_particle_costs)

    # PSO hyperparameters
    w = 0.5    # inertia
    c1 = 1.5   # cognitive
    c2 = 1.5   # social

    cost_history = []

    # PSO main loop
    for iteration in range(num_iterations):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (best_particle_positions[i] - particles[i])
                + c2 * r2 * (global_best_position - particles[i])
            )

            # Update position
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)  # Bound search space

            # Evaluate new cost
            reshaped = particles[i].reshape(-1, 3)
            cost = objective_function(reshaped)

            # Update personal best
            if cost < best_particle_costs[i]:
                best_particle_costs[i] = cost
                best_particle_positions[i] = particles[i]

            # Update global best
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particles[i]

        cost_history.append(global_best_cost)

    return global_best_position.reshape(-1, 3), cost_history

# ---------------------------------------------
# Visualization: 4 Optimized Antennas
# ---------------------------------------------
fig = plt.figure(figsize=(20, 10))

for i in range(4):
    best_antenna_points, cost_history = run_pso()

    # 3D Antenna Plot
    ax = fig.add_subplot(2, 4, i + 1, projection='3d')
    ax.plot(best_antenna_points[:, 0], best_antenna_points[:, 1], best_antenna_points[:, 2],
            marker='o', linewidth=2)
    ax.set_title(f"Antenna Design {i + 1}", fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.scatter(*best_antenna_points[0], color='red', s=100, label='Start')
    ax.scatter(*best_antenna_points[-1], color='green', s=100, label='End')
    ax.legend(fontsize=10)

    # Cost Over Iterations
    ax2 = fig.add_subplot(2, 4, i + 5)
    ax2.plot(range(1, num_iterations + 1), cost_history,
             marker='o', markersize=3, color='blue', linewidth=2)
    ax2.set_title(f"Cost Over Iterations {i + 1}", fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Cost", fontsize=12)
    ax2.grid(True)

plt.suptitle(" Optimized Antenna Structures Using Particle Swarm Optimization", fontsize=18)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
