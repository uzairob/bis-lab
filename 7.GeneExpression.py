import numpy as np
import random

# Define the Rastrigin function (objective function)
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Initialize population
def initialize_population(pop_size, gene_length, search_range):
    return [np.random.uniform(search_range[0], search_range[1], gene_length) for _ in range(pop_size)]

# Evaluate fitness
def evaluate_fitness(population):
    return [rastrigin_function(ind) for ind in population]

# Selection (tournament selection)
def selection(population, fitness):
    selected = []
    for _ in range(len(population)):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if fitness[i] < fitness[j] else population[j])
    return selected

# Crossover (uniform crossover)
def crossover(parent1, parent2):
    child = []
    for p1, p2 in zip(parent1, parent2):
        child.append(p1 if random.random() < 0.5 else p2)
    return np.array(child)

# Mutation (random mutation)
def mutate(individual, mutation_rate, search_range):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(search_range[0], search_range[1])
    return individual

# Gene Expression Algorithm
def gene_expression_algorithm(pop_size, gene_length, generations, mutation_rate, search_range):
    # Step 1: Initialize population
    population = initialize_population(pop_size, gene_length, search_range)

    for generation in range(generations):
        # Step 2: Evaluate fitness
        fitness = evaluate_fitness(population)

        # Step 3: Selection
        selected_population = selection(population, fitness)

        # Step 4: Crossover and Mutation
        next_population = []
        for i in range(0, len(selected_population), 2):
            if i + 1 < len(selected_population):
                # Crossover
                child1 = crossover(selected_population[i], selected_population[i + 1])
                child2 = crossover(selected_population[i + 1], selected_population[i])
            else:
                child1 = selected_population[i]
                child2 = selected_population[i]
            # Mutation
            next_population.append(mutate(child1, mutation_rate, search_range))
            next_population.append(mutate(child2, mutation_rate, search_range))

        population = next_population[:pop_size]  # Maintain population size

    # Final fitness evaluation
    fitness = evaluate_fitness(population)
    best_individual = population[np.argmin(fitness)]
    return best_individual, rastrigin_function(best_individual)

# Parameters
pop_size = 50
gene_length = 10
generations = 100
mutation_rate = 0.1
search_range = (-5.12, 5.12)  # Search range for Rastrigin function

# Run the algorithm
best_solution, best_fitness = gene_expression_algorithm(pop_size, gene_length, generations, mutation_rate, search_range)

print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")
