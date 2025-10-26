import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import random
import warnings

warnings.filterwarnings("ignore")

# === 8 selected benchmark functions ===
def sphere(x):
    x = np.array(x)
    return np.sum(x**2)

def matyas(x):
    x = np.array(x)
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def rosenbrock(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def powell(x):
    x = np.array(x)
    term1 = (x[0] + 10*x[1])**2
    term2 = 5 * (x[2] - x[3])**2
    term3 = (x[1] - 2*x[2])**4
    term4 = 10 * (x[0] - x[3])**4
    return term1 + term2 + term3 + term4

def rastrigin(x):
    x = np.array(x)
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    x = np.array(x)
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)

def eggholder(x):
    x = np.array(x)
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))

def beale(x):
    x = np.array(x)
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


# === Differential Evolution Algorithm ===
def differential_evolution(func, bounds, population_size=50, generations=100, F=0.5, CR=0.7):
    dimensions = len(bounds)
    population = [np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], dimensions) for _ in range(population_size)]
    costs = []

    for generation in range(generations):
        for i in range(population_size):
            indices = list(range(population_size))
            indices.remove(i)
            a, b, c = random.sample(indices, 3)
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])
            trial = np.where(np.random.rand(dimensions) < CR, mutant, population[i])
            if func(trial) < func(population[i]):
                population[i] = trial

        best = min(population, key=func)
        costs.append(func(best))

    return best, costs


# === Function groups (8 only) ===
functions = [
    ("Sphere (Convex)", sphere, [(-5, 5)] * 2),
    ("Matyas (Convex)", matyas, [(-10, 10)] * 2),
    ("Rosenbrock (Non-convex)", rosenbrock, [(-5, 5)] * 2),
    ("Powell (Non-convex)", powell, [(-5, 5)] * 4),
    ("Rastrigin (Multimodal)", rastrigin, [(-5, 5)] * 2),
    ("Ackley (Multimodal)", ackley, [(-5, 5)] * 2),
    ("Eggholder (Rugged)", eggholder, [(-512, 512)] * 2),
    ("Beale (Rugged)", beale, [(-4.5, 4.5)] * 2)
]



# === Run optimization and store results ===
results = {}

for name, func, bounds in functions:
    print(f"\nRunning {name}...")

    start_time = time.time()
    memory_before = memory_usage()[0]

    best_x, costs = differential_evolution(func, bounds, population_size=50, generations=100)

    memory_after = memory_usage()[0]
    end_time = time.time()

    print(f"Best Cost: {costs[-1]:.6f}")
    print(f"Time: {end_time - start_time:.4f} sec")
    print(f"Memory: {memory_after - memory_before:.4f} MB")

    results[name] = {
        "func": func,
        "bounds": bounds,
        "best_x": best_x,
        "costs": costs
    }

# === Plot 1: Convergence curves ===
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()

for idx, (name, res) in enumerate(results.items()):
    axes[idx].plot(res["costs"], linewidth=2, color="darkblue")
    axes[idx].set_title(name, fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Generations", fontsize=10)
    axes[idx].set_ylabel("Cost", fontsize=10)
    axes[idx].grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Convergence Curves of Selected Benchmark Functions", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()



# === Plot 2: Landscapes with best solution ===
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.ravel()

for idx, (name, res) in enumerate(results.items()):
    func, bounds, best_x = res["func"], res["bounds"], res["best_x"]

    if len(bounds) == 2:  # 2D functions only
        x = np.linspace(bounds[0][0], bounds[0][1], 200)
        y = np.linspace(bounds[1][0], bounds[1][1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[func([xx, yy]) for xx, yy in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

        cp = axes[idx].contourf(X, Y, Z, levels=50, cmap="viridis")
        axes[idx].scatter(best_x[0], best_x[1], color="red", marker="*", s=120, label="Best")
        axes[idx].set_title(name, fontsize=14, fontweight="bold")
        axes[idx].legend()
    else:
        axes[idx].text(0.5, 0.5, "High-dim\n(No plot)", ha="center", va="center", fontsize=12)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")

plt.suptitle("Function Landscapes with Best Solution Found", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
