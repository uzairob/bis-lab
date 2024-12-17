import numpy as np

# Define the objective function
def objective_function(x):
    return x**2 - 4*x + 4

# Initialize the grid
def initialize_grid(grid_size, search_range):
    return np.random.uniform(search_range[0], search_range[1], (grid_size, grid_size))

# Compute fitness for the grid
def evaluate_fitness(grid, objective_function):
    return objective_function(grid)

# Update the grid based on neighborhood average
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Get neighbors' values
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                        neighbors.append(grid[ni, nj])
            # Update state to the average of neighbors
            new_grid[i, j] = np.mean(neighbors)
    return new_grid

# Main function to run the algorithm
def parallel_cellular_algorithm(grid_size, search_range, iterations):
    grid = initialize_grid(grid_size, search_range)  # Step 2: Initialize grid
    for _ in range(iterations):
        fitness = evaluate_fitness(grid, objective_function)  # Step 3: Evaluate fitness
        grid = update_grid(grid)  # Step 4: Update states
    # Find the best solution
    best_value = grid[np.unravel_index(np.argmin(fitness), fitness.shape)]
    return best_value, objective_function(best_value)

# Parameters
grid_size = 10  # 10x10 grid
search_range = (-10, 10)  # Search range for cell values
iterations = 100  # Number of iterations

# Run the algorithm
best_value, best_fitness = parallel_cellular_algorithm(grid_size, search_range, iterations)

# Output the results
print(f"Best Value: {best_value}")
print(f"Best Fitness: {best_fitness}")
