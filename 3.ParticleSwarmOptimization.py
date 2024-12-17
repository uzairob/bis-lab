import numpy as np

# Objective function (Example: Rastrigin function)
def objective_function(position):
    return sum([x**2 - 10 * np.cos(2 * np.pi * x) + 10 for x in position])

# Particle Swarm Optimization
class Particle:
    def __init__(self, dimensions):
        self.position = np.random.uniform(-10, 10, dimensions)  # Initialize position
        self.velocity = np.random.uniform(-1, 1, dimensions)    # Initialize velocity
        self.best_position = self.position.copy()               # Personal best position
        self.best_score = float('inf')                          # Best score for personal best

    def update_velocity(self, global_best_position, inertia, cognitive_const, social_const):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = cognitive_const * r1 * (self.best_position - self.position)
        social = social_const * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity

# PSO Algorithm
def particle_swarm_optimization(objective_func, dimensions, num_particles, max_iter):
    inertia = 0.5            # Inertia weight
    cognitive_const = 1.5    # Cognitive constant
    social_const = 1.5       # Social constant

    # Initialize particles
    swarm = [Particle(dimensions) for _ in range(num_particles)]
    global_best_position = np.random.uniform(-10, 10, dimensions)
    global_best_score = float('inf')

    for iteration in range(max_iter):
        for particle in swarm:
            # Evaluate fitness
            fitness = objective_func(particle.position)
            # Update personal best
            if fitness < particle.best_score:
                particle.best_score = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particle.position.copy()

        # Update velocity and position for each particle
        for particle in swarm:
            particle.update_velocity(global_best_position, inertia, cognitive_const, social_const)
            particle.update_position()

        print(f"Iteration {iteration+1}/{max_iter}, Global Best Score: {global_best_score}")

    return global_best_position, global_best_score

# Example usage
best_position, best_score = particle_swarm_optimization(objective_function, dimensions=2, num_particles=30, max_iter=100)
print("Best Position:", best_position)
print("Best Score:", best_score)
