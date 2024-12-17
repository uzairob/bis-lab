import random

# Define the fitness function
def fitness_function(x):
    return x**2  # Example function: f(x) = x^2

# Generate initial population
def generate_population(size, x_min, x_max):
    return [random.uniform(x_min, x_max) for _ in range(size)]

# Selection process
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    parents = random.choices(population, weights=selection_probs, k=2)
    return parents

# Crossover process
def crossover(parent1, parent2):
    alpha = random.random()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child

# Mutation process
def mutate(child, mutation_rate, x_min, x_max):
    if random.random() < mutation_rate:
        child = random.uniform(x_min, x_max)
    return child

# Genetic Algorithm
def genetic_algorithm(pop_size, generations, mutation_rate, x_min, x_max):
    population = generate_population(pop_size, x_min, x_max)
    for generation in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, x_min, x_max)
            new_population.append(child)
        population = new_population
    best_solution = max(population, key=fitness_function)
    return best_solution

# User inputs
pop_size = int(input("Enter population size: "))
generations = int(input("Enter number of generations: "))
mutation_rate = float(input("Enter mutation rate (0-1): "))
x_min = float(input("Enter minimum value of x: "))
x_max = float(input("Enter maximum value of x: "))

# Run the genetic algorithm
best_solution = genetic_algorithm(pop_size, generations, mutation_rate, x_min, x_max)
print(f"The best solution found is: {best_solution} with fitness value: {fitness_function(best_solution)}")
