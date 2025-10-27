import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import random
import warnings

warnings.filterwarnings("ignore")

# === Genetic Algorithm (GA) ===
def genetic_algorithm(func, bounds, population_size=50, generations=100,
                      mutation_rate=0.1, crossover_rate=0.7):
    dimensions = len(bounds)
    population = [np.array([random.uniform(b[0], b[1]) for b in bounds])
                  for _ in range(population_size)]

    def mutate(ind):
        for i in range(dimensions):
            if random.random() < mutation_rate:
                ind[i] = random.uniform(bounds[i][0], bounds[i][1])
        return ind

    def crossover(p1, p2):
        if random.random() < crossover_rate:
            point = random.randint(1, dimensions - 1)
            return np.concatenate((p1[:point], p2[point:]))
        return p1

    def select(population, fitness):
        min_fit = min(fitness)
        shifted = [f - min_fit + 1e-6 for f in fitness]
        total = sum(shifted)
        probs = [f / total for f in shifted]
        return population[np.random.choice(len(population), p=probs)]

    best_individual, best_cost = None, float('inf')
    costs = []

    for _ in range(generations):
        fitness = [-func(ind) for ind in population]
        next_population = []
        for _ in range(population_size):
            p1, p2 = select(population, fitness), select(population, fitness)
            child = mutate(crossover(p1, p2))
            next_population.append(child)
        population = next_population
        generation_best = min(population, key=func)
        gen_cost = func(generation_best)
        if gen_cost < best_cost:
            best_individual, best_cost = generation_best, gen_cost
        costs.append(best_cost)

    return best_individual, costs


# === 8 Selected Benchmark Functions ===
def sphere(x):
    x = np.array(x); return np.sum(x**2)

def matyas(x):
    x = np.array(x); return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

def rosenbrock(x):
    x = np.array(x); return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

def powell(x):
    x = np.array(x)
    return (x[0]+10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4

def rastrigin(x):
    x = np.array(x); A = 10
    return A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))

def ackley(x):
    x = np.array(x); a,b,c = 20,0.2,2*np.pi; d=len(x)
    return -a*np.exp(-b*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(c*x))/d) + a + np.exp(1)

def eggholder(x):
    x = np.array(x)
    return -(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2+(x[1]+47)))) - x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47))))

def beale(x):
    x = np.array(x)
    return (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2


# === Functions (8 only) ===
functions = [
    ("Sphere (Convex)", sphere, [(-5, 5)]*2),
    ("Matyas (Convex)", matyas, [(-10, 10)]*2),
    ("Rosenbrock (Non-convex)", rosenbrock, [(-5, 5)]*2),
    ("Powell (Non-convex)", powell, [(-5, 5)]*4),
    ("Rastrigin (Multimodal)", rastrigin, [(-5, 5)]*2),
    ("Ackley (Multimodal)", ackley, [(-5, 5)]*2),
    ("Eggholder (Rugged)", eggholder, [(-512, 512)]*2),
    ("Beale (Rugged)", beale, [(-4.5, 4.5)]*2)
]

# === Run optimization and store results ===
results = {}
for name, func, bounds in functions:
    print(f"\nRunning {name}...")
    start_time = time.time(); memory_before = memory_usage()[0]
    best_x, costs = genetic_algorithm(func, bounds, population_size=50, generations=100)
    memory_after = memory_usage()[0]; end_time = time.time()

    print(f"Best Cost: {costs[-1]:.6f}")
    print(f"Time: {end_time-start_time:.4f} sec")
    print(f"Memory: {memory_after-memory_before:.4f} MB")

    results[name] = {"func": func, "bounds": bounds, "best_x": best_x, "costs": costs}

# === Plot 1: Convergence curves ===
fig, axes = plt.subplots(2, 4, figsize=(22, 10)); axes = axes.ravel()
for idx, (name, res) in enumerate(results.items()):
    axes[idx].plot(res["costs"], linewidth=2, color="darkblue")
    axes[idx].set_title(name, fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Generations", fontsize=10)
    axes[idx].set_ylabel("Cost", fontsize=10)
    axes[idx].grid(True, linestyle="--", alpha=0.6)
plt.suptitle("Convergence Curves of Selected Benchmark Functions (GA)", fontsize=16, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97]); plt.show()

# === Plot 2: Landscapes with best solution ===
fig, axes = plt.subplots(2, 4, figsize=(22, 10)); axes = axes.ravel()
for idx, (name, res) in enumerate(results.items()):
    func,bounds,best_x = res["func"],res["bounds"],res["best_x"]
    if len(bounds)==2:
        x = np.linspace(bounds[0][0], bounds[0][1], 200)
        y = np.linspace(bounds[1][0], bounds[1][1], 200)
        X,Y = np.meshgrid(x,y)
        Z = np.array([[func([xx,yy]) for xx,yy in zip(rx,ry)] for rx,ry in zip(X,Y)])
        cp = axes[idx].contourf(X,Y,Z,levels=50,cmap="viridis")
        axes[idx].scatter(best_x[0],best_x[1],color="red",marker="*",s=120,label="Best")
        axes[idx].set_title(name, fontsize=14, fontweight="bold"); axes[idx].legend()
    else:
        axes[idx].text(0.5,0.5,"High-dim\n(No plot)",ha="center",va="center",fontsize=12)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
plt.suptitle("Function Landscapes with Best Solution Found (GA)", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97]); plt.show()
