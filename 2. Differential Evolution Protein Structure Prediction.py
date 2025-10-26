# ---------------------------------------------
# Import Required Libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform # to compute pairwise distances
from scipy.interpolate import splprep, splev # for 3D smoothing curve interpolation
from mpl_toolkits.mplot3d import Axes3D # for 3D protein structure visualization
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------
# Energy Function for Protein Model
# ---------------------------------------------
def energy_function(positions):
    """
    Compute the total potential energy of the protein structure.
    Combines Lennard-Jones (non-adjacent) + harmonic bond energy (adjacent).
    """
    distances = pdist(positions)
    distances = np.clip(distances, 1e-6, None)  # Prevent division by zero

    # Lennard-Jones potential (non-bonded residues)
    lj_energy = np.sum(4 * ((1 / distances)**12 - (1 / distances)**6))

    # Harmonic bond energy for adjacent residues
    distances_matrix = squareform(distances)
    bond_indices = np.arange(len(positions) - 1)
    bond_dists = distances_matrix[bond_indices, bond_indices + 1]
    bond_energy = 0.5 * np.sum((bond_dists - 1.0)**2)

    return lj_energy + bond_energy

# ---------------------------------------------
#  DE Hyperparameters & Initialization
# ---------------------------------------------
num_particles = 40               # Number of individuals in the population
num_amino_acids = 15             # Protein size (residues)
num_dimensions = 3               # 3D space
num_iterations = 1000            # Evolutionary generations
mutation_factor = 0.5            # Differential mutation scale
crossover_probability = 0.9      # Crossover rate

# Initialize particles in 3D space: shape [particles, residues, dimensions]
positions = np.random.uniform(-5, 5, (num_particles, num_amino_acids, num_dimensions))
fitness_scores = np.array([energy_function(p) for p in positions])  # Initial fitness

convergence = []  # Store best score per iteration

# ---------------------------------------------
# Differential Evolution Main Loop
# ---------------------------------------------
print(" Starting Differential Evolution...")
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Mutation
        candidates = np.delete(np.arange(num_particles), i)
        a, b, c = positions[np.random.choice(candidates, 3, replace=False)]
        donor_vector = a + mutation_factor * (b - c)

        # Crossover
        trial_vector = np.copy(positions[i])
        for j in range(num_amino_acids):
            if np.random.rand() < crossover_probability:
                trial_vector[j] = donor_vector[j]

        # Evaluate trial
        trial_fitness = energy_function(trial_vector)
        if trial_fitness < fitness_scores[i]:
            positions[i] = trial_vector
            fitness_scores[i] = trial_fitness

    # Track global best
    best_idx = np.argmin(fitness_scores)
    best_pos = positions[best_idx]
    best_score = fitness_scores[best_idx]
    convergence.append(best_score)

    # Progress print
    print(f"Iteration {iteration + 1:03d}/{num_iterations} | Best Energy: {best_score:.4f}")

# ---------------------------------------------
# Plot Convergence Curve
# ---------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(convergence, marker='o', markersize=2, linewidth=2, color='blue')
plt.title(" Convergence of DE on Protein Energy Minimization", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Best Energy", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# Visualize Optimized Protein Structure (3D)
# ---------------------------------------------
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot amino acids
ax.scatter(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2],
           c='red', s=100, label='Amino Acids')

# blue backbone curve pass through all amino acids
tck, _ = splprep([best_pos[:, 0], best_pos[:, 1], best_pos[:, 2]], s=0)  # <-- s=0 is key!
smooth = splev(np.linspace(0, 1, 300), tck)

# Plot connected blue curve 
ax.plot(smooth[0], smooth[1], smooth[2],
        color='blue', linewidth=3, label='Backbone Curve')

# Annotate amino acid indices
for i, (x, y, z) in enumerate(best_pos):
    ax.text(x, y, z, str(i), fontsize=9, color='black')

# Labels and layout
ax.set_title("Smooth & Connected Protein Backbone", fontsize=14)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------
# Final Results Summary
# ---------------------------------------------
print("Final Optimized Amino Acid Positions:\n", best_pos)
print(f"Final Energy: {best_score:.4f}")
