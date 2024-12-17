import numpy as np

# Objective function (e.g., Sphere function)
def objective_function(position):
    return sum(x**2 for x in position)

# Grey Wolf Optimizer
def grey_wolf_optimizer(obj_function, dim, pop_size, max_iter, bounds=(-10, 10)):
    a = 2  # Coefficient, decreases linearly from 2 to 0
    alpha_position = np.zeros(dim)
    alpha_score = float('inf')  # Best fitness (alpha)
    beta_position = np.zeros(dim)
    beta_score = float('inf')  # Second-best fitness (beta)
    delta_position = np.zeros(dim)
    delta_score = float('inf')  # Third-best fitness (delta)

    # Initialize the positions of the wolves
    wolves = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

    for iteration in range(max_iter):
        for i, wolf in enumerate(wolves):
            fitness = obj_function(wolf)

            # Update alpha, beta, and delta
            if fitness < alpha_score:
                delta_position = beta_position.copy()
                delta_score = beta_score
                beta_position = alpha_position.copy()
                beta_score = alpha_score
                alpha_position = wolf.copy()
                alpha_score = fitness
            elif fitness < beta_score:
                delta_position = beta_position.copy()
                delta_score = beta_score
                beta_position = wolf.copy()
                beta_score = fitness
            elif fitness < delta_score:
                delta_position = wolf.copy()
                delta_score = fitness

        # Update positions
        for i, wolf in enumerate(wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_position - wolf)
            X1 = alpha_position - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_position - wolf)
            X2 = beta_position - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_position - wolf)
            X3 = delta_position - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3

        # Linearly decrease a
        a -= 2 / max_iter

        print(f"Iteration {iteration+1}/{max_iter}, Alpha Score: {alpha_score}")

    return alpha_position, alpha_score

# Example usage
best_position, best_score = grey_wolf_optimizer(objective_function, dim=2, pop_size=30, max_iter=100)
print("Best Position:", best_position)
print("Best Score:", best_score)
