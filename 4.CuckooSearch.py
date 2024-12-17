import numpy as np

# Objective function (example: Sphere function)
def objective_function(x):
    return np.sum(x ** 2)

# Levy flight implementation
def levy_flight(Lambda, dim, alpha=1.0):
    u = np.random.normal(0, 1, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = alpha * (u / (np.abs(v) ** (1 / Lambda)))  # Lévy step
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n, max_generations, pa, lower_bound, upper_bound, dim):
    # Step 1: Initialize nests randomly
    nests = np.random.uniform(lower_bound, upper_bound, size=(n, dim))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    # Iterative optimization
    for t in range(max_generations):
        # Rule 1: Generate new solutions via Lévy flight
        for i in range(n):
            new_nest = nests[i] + levy_flight(1.5, dim)
            new_nest = np.clip(new_nest, lower_bound, upper_bound)
            new_fitness = objective_function(new_nest)

            # Rule 2: Replace nests if better
            if new_fitness < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

                # Update global best
                if new_fitness < best_fitness:
                    best_nest = new_nest
                    best_fitness = new_fitness

        # Rule 3: Abandon some nests and create new random ones
        abandon = np.random.rand(n) < pa
        nests[abandon] = np.random.uniform(lower_bound, upper_bound, size=(np.sum(abandon), dim))
        fitness[abandon] = np.array([objective_function(nest) for nest in nests[abandon]])

    return best_nest, best_fitness

# Parameters
n = 25  # Number of nests
dim = 5  # Dimensionality of the problem
max_generations = 100  # Max iterations
pa = 0.25  # Abandonment probability
lower_bound = -10  # Lower bound of the search space
upper_bound = 10  # Upper bound of the search space

# Run Cuckoo Search
best_solution, best_value = cuckoo_search(n, max_generations, pa, lower_bound, upper_bound, dim)

print("Best solution found:", best_solution)
print("Best objective value:", best_value)
