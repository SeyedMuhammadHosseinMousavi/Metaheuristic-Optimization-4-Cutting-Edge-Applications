import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time
import random
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

    costs = []

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
        costs.append(global_best_score)

    return global_best_position, costs


# === 8 Selected Benchmark Functions ===
def sphere(x): return np.sum(np.array(x)**2)

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
    return -(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2+(x[1]+47)))) \
           - x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47))))

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
    best_x, costs = particle_swarm_optimization(func, bounds, swarm_size=50, generations=100)
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
plt.suptitle("Convergence Curves of Selected Benchmark Functions (PSO)", fontsize=16, fontweight="bold")
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
plt.suptitle("Function Landscapes with Best Solution Found (PSO)", fontsize=20, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97]); plt.show()
