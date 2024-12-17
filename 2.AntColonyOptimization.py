import numpy as np
import random

class AntColony:
    def __init__(self, distance_matrix, n_ants, n_iterations, decay, alpha=1, beta=1):
        self.distance_matrix = distance_matrix
        self.pheromone = np.ones(distance_matrix.shape) / len(distance_matrix)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.all_indices = range(len(distance_matrix))

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("path", np.inf)

        for _ in range(self.n_iterations):
            all_paths = self.generate_all_paths()
            self.update_pheromones(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path

        return all_time_shortest_path

    def generate_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.generate_path(0)  # Start from city 0
            path_dist = self.calculate_path_distance(path)
            all_paths.append((path, path_dist))
        return all_paths

    def generate_path(self, start):
        path = [start]
        visited = set(path)
        while len(visited) < len(self.distance_matrix):
            move = self.select_next_city(path[-1], visited)
            path.append(move)
            visited.add(move)
        path.append(start)  # Return to starting city
        return path

    def select_next_city(self, current_city, visited):
        pheromone = np.copy(self.pheromone[current_city])
        pheromone[list(visited)] = 0  # Avoid visiting already visited cities

        probabilities = pheromone ** self.alpha * ((1 / self.distance_matrix[current_city]) ** self.beta)
        probabilities /= probabilities.sum()  # Normalize probabilities

        next_city = np.random.choice(self.all_indices, p=probabilities)
        return next_city

    def calculate_path_distance(self, path):
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distance_matrix[path[i]][path[i + 1]]
        return total_dist

    def update_pheromones(self, all_paths):
        self.pheromone *= (1 - self.decay)  # Pheromone evaporation
        for path, dist in all_paths:
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += 1 / dist  # Update pheromone based on path quality

# Example: A 4-city TSP problem
if __name__ == "__main__":
    distance_matrix = np.array([[np.inf, 12, 12, 15],
                                [12, np.inf, 13, 14],
                                [12, 13, np.inf, 11],
                                [15, 14, 11, np.inf]])

    colony = AntColony(distance_matrix, n_ants=10, n_iterations=100, decay=0.1, alpha=1, beta=2)
    best_path = colony.run()
    print("Best path found:", best_path)
