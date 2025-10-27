import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from memory_profiler import memory_usage
import warnings

warnings.filterwarnings("ignore")

# === Particle Swarm Optimization (PSO) ===
def particle_swarm_optimization(func, bounds, swarm_size=50, generations=100,
                                inertia=0.5, cognitive=1.5, social=1.5):
    dimensions = len(bounds)
    particles = [np.random.uniform([b[0] for b in bounds],
                                   [b[1] for b in bounds],
                                   dimensions) for _ in range(swarm_size)]
    velocities = [np.random.uniform(-1, 1, dimensions) for _ in range(swarm_size)]
    personal_best_positions = particles[:]
    personal_best_scores = [func(p) for p in particles]
    global_best_position = particles[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    for _ in range(generations):
        for i, particle in enumerate(particles):
            velocities[i] = (inertia * velocities[i]
                             + cognitive * random.random() * (personal_best_positions[i] - particle)
                             + social * random.random() * (global_best_position - particle))
            particles[i] = np.clip(particle + velocities[i],
                                   [b[0] for b in bounds],
                                   [b[1] for b in bounds])
            score = func(particles[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        global_best_position = particles[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

    return global_best_position, global_best_score


# === Functions (4 only) ===
def sphere(x): return np.sum(np.array(x)**2)

def rosenbrock(x):
    x = np.array(x); return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

def ackley(x):
    x = np.array(x); a,b,c = 20,0.2,2*np.pi; d=len(x)
    return -a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.exp(1)

def eggholder(x):
    x = np.array(x)
    return -(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2+(x[1]+47)))) \
           - x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47))))


# === Selected Functions with bounds ===
functions = [
    ("Sphere", sphere, [(-5, 5)]*2),
    ("Rosenbrock", rosenbrock, [(-5, 5)]*2),
    ("Ackley", ackley, [(-5, 5)]*2),
    ("Eggholder", eggholder, [(-512, 512)]*2)
]

# === Run optimization and plot 3D landscapes ===
results = {}
for name, func, bounds in functions:
    print(f"\nRunning {name}...")
    start_time = time.time(); memory_before = memory_usage()[0]
    best_x, best_cost = particle_swarm_optimization(func, bounds, swarm_size=50, generations=100)
    memory_after = memory_usage()[0]; end_time = time.time()

    print(f"Best Cost: {best_cost:.6f}")
    print(f"Time: {end_time-start_time:.4f} sec")
    print(f"Memory: {memory_after-memory_before:.4f} MB")

    results[name] = {"func": func, "bounds": bounds, "best_x": best_x, "best_cost": best_cost}

# === Plot 3D Landscapes (2x2) ===
fig = plt.figure(figsize=(16, 12))

for idx, (name, res) in enumerate(results.items(), 1):
    func, bounds, best_x = res["func"], res["bounds"], res["best_x"]
    ax = fig.add_subplot(2, 2, idx, projection='3d')

    # Mesh grid
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([xx, yy]) for xx, yy in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    # Surface plot
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True)
    ax.scatter(best_x[0], best_x[1], func(best_x), color="red", marker="*", s=120, label="Best")
    ax.set_title(name, fontsize=14, fontweight="bold")
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Cost", fontsize=10)
    ax.legend()

plt.suptitle("3D Landscapes with Best Solution Found (PSO)", fontsize=18, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
