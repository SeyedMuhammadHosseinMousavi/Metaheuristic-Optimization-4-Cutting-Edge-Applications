# ---------------------------------------------
# Import Required Libraries
# ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
# np.random.seed(42)

# ---------------------------------------------
# Generate Random Exoplanet Environment
# ---------------------------------------------
def generate_planet():
    """Generate parameters for a random exoplanetary environment."""
    planet_name = f"Planet_{np.random.randint(1000, 9999)}"
    star_name = f"Star_{np.random.randint(1000, 9999)}"

    return {
        "planet_name": planet_name,
        "star_name": star_name,
        "gravity": np.random.uniform(0.1, 3.0),  # Earth G
        "atmosphere_composition": {
            "O2": np.random.uniform(0.01, 0.5),
            "CO2": np.random.uniform(0.01, 0.5),
            "Other Gases": np.random.uniform(0.01, 0.9),
        },
        "radiation_level": np.random.uniform(1, 500),  # mSv/year
        "temperature_range": (
            np.random.uniform(-100, 0),
            np.random.uniform(0, 100)
        ),
        "day_length": np.random.uniform(6, 48),  # hours
    }

# ---------------------------------------------
# Fitness Function: Evaluate Genetic Profile
# ---------------------------------------------
def objective_function(genetic_profile, planet_params):
    """
    Score a genetic profile based on survivability in given planet conditions.
    Traits: radiation_resistance, bone_density, oxygen_efficiency,
            temp_adaptability, stress_resilience
    """
    gravity = planet_params["gravity"]
    atmosphere = planet_params["atmosphere_composition"]
    radiation = planet_params["radiation_level"]
    temp_min, temp_max = planet_params["temperature_range"]
    day_length = planet_params["day_length"]

    radiation_resistance, bone_density, oxygen_efficiency, temp_adaptability, stress_resilience = genetic_profile

    fitness_radiation = np.exp(-radiation / radiation_resistance)
    fitness_gravity = np.exp(-abs(gravity - 1) / bone_density)
    fitness_oxygen = oxygen_efficiency * atmosphere["O2"]
    avg_temp = (temp_min + temp_max) / 2
    fitness_temperature = np.exp(-abs(avg_temp) / temp_adaptability)
    fitness_stress = stress_resilience / day_length

    # Weighted fitness components
    return (0.25 * fitness_radiation +
            0.2  * fitness_gravity +
            0.25 * fitness_oxygen +
            0.2  * fitness_temperature +
            0.1  * fitness_stress)

# ---------------------------------------------
# Genetic Algorithm Parameters
# ---------------------------------------------
population_size = 500
num_generations = 1000
num_genes = 5  # Number of traits
mutation_rate = 0.2

trait_bounds = (0.5, 5.0)  # Reasonable range for genetic traits

# Initialize population: each row = [r_resist, b_density, o2_eff, temp_adapt, stress_res]
population = np.random.uniform(*trait_bounds, size=(population_size, num_genes))

# Generate random planet
planet_params = generate_planet()
fitness_history = []


# ---------------------------------------------
# Evolutionary Loop
# ---------------------------------------------
print("Starting Genetic Algorithm for Exoplanet Survival...\n")

for generation in range(num_generations):
    # Evaluate population fitness
    fitness = np.array([objective_function(ind, planet_params) for ind in population])
    fitness_history.append(np.max(fitness))
    best_idx = np.argmax(fitness)
    best_individual = population[best_idx]

    print(f"Generation {generation + 1:03d} | Best Fitness: {fitness[best_idx]:.4f}")

    # Selection (Roulette Wheel)
    probabilities = fitness / np.sum(fitness)
    selected_indices = np.random.choice(np.arange(population_size), size=population_size, p=probabilities)
    selected_population = population[selected_indices]

    # Crossover (Single-point)
    new_population = []
    for i in range(0, population_size, 2):
        parent1 = selected_population[i]
        parent2 = selected_population[(i + 1) % population_size]
        cp = np.random.randint(1, num_genes)
        child1 = np.concatenate([parent1[:cp], parent2[cp:]])
        child2 = np.concatenate([parent2[:cp], parent1[cp:]])
        new_population.extend([child1, child2])

    # Mutation (Gaussian + Clipping)
    new_population = np.array(new_population)
    mutation_mask = np.random.rand(population_size, num_genes) < mutation_rate
    new_population[mutation_mask] += np.random.normal(0, 0.1, size=mutation_mask.sum())
    new_population = np.clip(new_population, *trait_bounds)  # Keep within biological limits

    # Update population
    population = new_population

# ---------------------------------------------
# Final Results
# ---------------------------------------------
final_fitness = np.array([objective_function(ind, planet_params) for ind in population])
best_idx = np.argmax(final_fitness)
best_individual = population[best_idx]
best_score = final_fitness[best_idx]

print("\nGA Optimization Completed!")
print("\n ")
print(f"0. Planet Name: {planet_params['planet_name']}")
print(f"0. Star Name: {planet_params['star_name']}")
print(f"1. Gravity: {planet_params['gravity']:.2f} G")
print("2. Atmosphere Composition:")
for gas, value in planet_params["atmosphere_composition"].items():
    print(f"  {gas}: {value * 100:.1f}%")
print(f"3. Radiation Level: {planet_params['radiation_level']:.1f} mSv/year")
print(f"4. Temperature Range: {planet_params['temperature_range'][0]:.1f}°C to {planet_params['temperature_range'][1]:.1f}°C")
print(f"5. Day Length: {planet_params['day_length']:.1f} hours")
best_generation = np.argmax(fitness_history) + 1
print(f"\nBest result achieved at Generation {best_generation} with Fitness {fitness_history[best_generation - 1]:.4f}")
print(f"\nBest Genetic Profile:\n{best_individual}")
best_idx = np.argmax(final_fitness)
best_individual = population[best_idx]
best_score = final_fitness[best_idx] 
print(f"Index in Population: {best_idx}")
print(f"Genetic Trait Values:")
print(f"  Radiation Resistance: {best_individual[0]:.4f}")
print(f"  Bone Density:         {best_individual[1]:.4f}")
print(f"  Oxygen Efficiency:    {best_individual[2]:.4f}")
print(f"  Temperature Adapt.:   {best_individual[3]:.4f}")
print(f"  Stress Resilience:    {best_individual[4]:.4f}")
print(f"\nBest Fitness Score: {best_score:.4f}")


# ---------------------------------------------
# Fitness Convergence Plot
# ---------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, linewidth=1, label="Best Fitness per Generation", color='darkgreen')
plt.title("Genetic Algorithm Optimization for Human Survival on Exoplanet", fontsize=14)
plt.xlabel("Generation", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
