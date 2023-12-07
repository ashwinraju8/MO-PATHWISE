import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from geomdl import NURBS
from geomdl import utilities as utils
from scipy.interpolate import interp1d
import multiprocessing
import multiprocessing.queues
import dill
from deap import base, creator, tools, algorithms
from functools import partial

import os
import pickle


def multiprocessing_setup():
    # Replace pickle with dill for serialization
    dill.extend(use_dill=True)

    # Create the multiprocessing pool
    pool = multiprocessing.Pool()
    return pool

class PathPlanningEnvironment:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.obstacles = []

    def create_random_obstacles(self, num_obstacles_range, obstacle_radius_range):
        num_obstacles = random.randint(*num_obstacles_range)
        self.obstacles = []
        for _ in range(num_obstacles):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            radius = random.uniform(*obstacle_radius_range)
            self.obstacles.append((x, y, radius))

    def set_fixed_obstacles(self, obstacle_data):
        # Setting fixed obstacles for debugging
        self.obstacles = obstacle_data

    def plot_environment(self):
        fig, ax = plt.subplots()
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red')
            ax.add_patch(circle)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', 'box')
        return fig, ax

    def convert_to_graph(self, buffer_size=1):
        G = nx.grid_2d_graph(int(self.width / self.resolution), int(self.height / self.resolution))
        # Assign a default weight of 1 to all edges
        for edge in G.edges():
            weight = random.uniform(1, 1.5)
            G[edge[0]][edge[1]]['weight'] = weight
        for obstacle in self.obstacles:
            ox, oy, oradius = obstacle
            for x in range(int(self.width / self.resolution)):
                for y in range(int(self.height / self.resolution)):
                    dx = x * self.resolution - ox
                    dy = y * self.resolution - oy
                    if np.sqrt(dx**2 + dy**2) <= oradius + buffer_size:
                        if (x, y) in G:
                            G.remove_node((x, y))
        return G

    def visualize_graph(self, G):
        pos = {(x, y): (x, y) for x, y in G.nodes()}
        nx.draw(G, pos, node_size=1, node_color='skyblue')

        # Plotting obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.gca().set_xlim(0, self.width)
        plt.gca().set_ylim(0, self.height)
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    def dijkstra_shortest_path(self, start, goal):
        G = self.convert_to_graph()

        try:
            path = nx.dijkstra_path(G, start, goal)
            return path
        except nx.NetworkXNoPath:
            print("No path found")
            return None
        
    def yen_k_shortest_paths(self, start, goal, K):
        # Initialize the list of shortest paths
        A = [self.dijkstra_shortest_path(start, goal)]
        if A[0] is None:
            return []

        B = []

        for k in range(1, K):
            for i in range(len(A[k - 1]) - 1):
                # Spur node is retrieved from the previous k-shortest path
                spur_node = A[k - 1][i]
                # The sequence of nodes from the source to the spur node
                root_path = A[k - 1][:i + 1]

                # Copy original graph
                G_original = self.convert_to_graph()
                G_spur = copy.deepcopy(G_original)

                for path in A:
                    if len(path) > i and root_path == path[:i + 1]:
                        # Remove the links that are part of the previous shortest paths which share the same root path
                        if (path[i], path[i + 1]) in G_spur.edges:
                            G_spur.remove_edge(path[i], path[i + 1])

                for node in root_path:
                    if node != spur_node:
                        G_spur.remove_node(node)

                try:
                    spur_path = nx.dijkstra_path(G_spur, spur_node, goal)
                except nx.NetworkXNoPath:
                    continue

                total_path = root_path[:-1] + spur_path
                if total_path not in B:
                    B.append(total_path)

            if not B:
                break

            # Sort the potential k-shortest paths by cost (distance)
            B.sort(key=lambda p: nx.path_weight(G_original, p, weight='weight'))
            # Add the lowest cost path becomes the k-shortest path
            A.append(B.pop(0))

        return A

    def generate_random_paths(self, start, goal, num_paths, num_control_points, noise_variance, max_attempts_per_point=10):
        random_paths = []

        for _ in range(num_paths):
            valid_path_found = False

            while not valid_path_found:
                line_points = np.linspace(np.array(start), np.array(goal), num_control_points)
                noisy_points = [start]

                for point in line_points[1:-1]:
                    valid_point_found = False
                    attempt_count = 0

                    while not valid_point_found and attempt_count < max_attempts_per_point:
                        noisy_point = tuple(point + np.random.normal(0, noise_variance, 2))
                        clamped_point = self.clamp_point_to_bounds(noisy_point)

                        if not self.is_point_in_obstacle(clamped_point):
                            noisy_points.append(clamped_point)
                            valid_point_found = True
                        else:
                            attempt_count += 1

                    if not valid_point_found:
                        break

                else:
                    noisy_points.append(goal)
                    random_paths.append(noisy_points)
                    valid_path_found = True

        return random_paths

    def is_point_within_bounds(self, point):
        x_min, y_min = 0, 0  # Assuming the minimum bounds are (0, 0)
        x_max, y_max = self.width, self.height
        x, y = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    def clamp_point_to_bounds(self, point):
        clamped_x = min(max(point[0], 0), self.width)
        clamped_y = min(max(point[1], 0), self.height)
        return clamped_x, clamped_y

    def is_point_in_obstacle(self, point):
        for ox, oy, radius in self.obstacles:
            if np.sqrt((point[0] - ox)**2 + (point[1] - oy)**2) <= radius:
                return True
        return False

    def is_line_intersecting_obstacles(self, start_point, end_point):
        for obstacle in self.obstacles:
            ox, oy, radius = obstacle
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            f = start_point[0] - ox, start_point[1] - oy
            a = dx * dx + dy * dy
            b = 2 * (f[0] * dx + f[1] * dy)
            c = (f[0] * f[0] + f[1] * f[1]) - radius * radius

            discriminant = b * b - 4 * a * c
            if discriminant >= 0:  # If discriminant is non-negative, lines may intersect
                discriminant = discriminant**0.5
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)
                if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                    return True  # Intersection occurs
        return False  # No intersection with any obstacles

    def convert_path_to_nurbs(self, path, standard_num_control_points=50, weights=None, is_grid_coords=True):
        if is_grid_coords:
            path = [(x * self.resolution, y * self.resolution) for x, y in path]

        current_num_control_points = len(path)
        
        if current_num_control_points != standard_num_control_points:
            path_array = np.array(path)
            distances = np.cumsum(np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)
            max_distance = distances[-1]
            target_distances = np.linspace(0, max_distance, standard_num_control_points)
            interp_func_x = interp1d(distances, path_array[:, 0], kind='linear', fill_value='extrapolate')
            interp_func_y = interp1d(distances, path_array[:, 1], kind='linear', fill_value='extrapolate')
            interpolated_x = interp_func_x(target_distances)
            interpolated_y = interp_func_y(target_distances)
            path = list(zip(interpolated_x, interpolated_y))

        # Adjust path to avoid obstacles and ensure within bounds
        adjusted_path = []
        for point in path:
            clamped_point = self.clamp_point_to_bounds(point)
            for ox, oy, radius in self.obstacles:
                if np.sqrt((clamped_point[0] - ox)**2 + (clamped_point[1] - oy)**2) <= 1.1*radius:
                    direction = np.array(clamped_point) - np.array([ox, oy])
                    direction /= np.linalg.norm(direction)
                    clamped_point = np.array([ox, oy]) + direction * (radius + self.resolution)
                    break
            adjusted_path.append(tuple(clamped_point))

        if weights is None:
            weights = [1] * len(adjusted_path)

        curve = NURBS.Curve()
        curve.degree = 3
        curve.ctrlpts = adjusted_path
        curve.weights = weights
        curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))

        return curve

    def plot_paths_on_environment(self, paths, is_nurbs=False):
        fig, ax = self.plot_environment()
        colors = list(plt.cm.tab10.colors)  # Predefined color set for distinction

        for idx, path in enumerate(paths):
            color = colors[idx % len(colors)]  # Cycle through colors

            if is_nurbs:
                # If the input is NURBS curves, plot the evaluated points
                curve_points = path.evalpts
                x_coords, y_coords = zip(*curve_points)
                # Scatter plot for the first and last points of the NURBS curve
                ax.scatter(x_coords[0], y_coords[0], color=color, marker='o')  # Start point
                ax.scatter(x_coords[-1], y_coords[-1], color=color, marker='o')  # End point
            else:
                # If the input is discretized paths, convert to environment coordinates
                path_env = [(x * self.resolution, y * self.resolution) for x, y in path]
                x_coords, y_coords = zip(*path_env)
                # Scatter plot for the first and last points of the discrete path
                ax.scatter(x_coords[0], y_coords[0], color=color, marker='o')  # Start point
                ax.scatter(x_coords[-1], y_coords[-1], color=color, marker='o')  # End point

            # Plot the path or NURBS curve
            ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f'Path {idx + 1}')

        ax.legend()
        plt.show()



class EvolutionaryPathPlanner:
    def __init__(self, path_planning_environment, cross_prob, mut_prob):
        self.env = path_planning_environment
        self.setup_deap()
        self.crossover_prob = cross_prob  # Example crossover probability
        self.mutation_prob = mut_prob   # Example mutation probability

    def setup_deap(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", self.breeding)
        self.toolbox.register("mutate", self.mutation, n_control_points=50)  
        self.toolbox.register("select", tools.selNSGA2)

    def breeding(self, ind1, ind2):
        # Extract control points and weights from each parent
        cp1, w1 = ind1
        cp2, w2 = ind2

        # Average control points (element-wise) and weights
        child_cp = [[(coord1 + coord2) / 2 for coord1, coord2 in zip(pt1, pt2)] for pt1, pt2 in zip(cp1, cp2)]
        child_weights = [(weight1 + weight2) / 2 for weight1, weight2 in zip(w1, w2)]

        # Construct and return the child individual
        return (child_cp, child_weights)

    def mutation(self, individual, n_control_points):
        D = 3 * n_control_points - 4  # Calculate D based on the number of control points
        tau0 = 1 / np.sqrt(2 * D)
        tau = 1 / np.sqrt(2 * np.sqrt(D))

        # Unpack the individual's control points and weights
        ctrlpts, weights = individual

        # Mutate control points
        mutated_cp = []
        for i, c in enumerate(ctrlpts):
            if i == 0 or i == len(ctrlpts) - 1:  # Keep endpoints fixed
                mutated_cp.append(c)
            else:
                step_size = np.exp(tau0 * np.random.normal() + tau * np.random.normal())
                mutated_cp.append([coord + step_size * np.random.normal(scale=0.5) for coord in c])

        # Mutate weights
        mutated_weights = [w + np.exp(tau0 * np.random.normal() + tau * np.random.normal()) * np.random.normal(scale=0.5) for w in weights]

        return [mutated_cp, mutated_weights]


    
    def calculate_curvature(self, first_derivative, second_derivative):
        v = np.array(first_derivative)  # Velocity vector (first derivative)
        v_prime = np.array(second_derivative)  # Acceleration vector (second derivative)

        v_norm = np.linalg.norm(v)
        T_prime = (v_prime * v_norm - v * np.dot(v, v_prime) / v_norm) / v_norm**2
        curvature = np.linalg.norm(T_prime) / v_norm**3

        return curvature

    def calculate_smoothness(self, curve, num_points=100):
        total_curvature = 0

        # Iterate over several points on the curve
        for i in range(num_points):
            u = i / float(num_points - 1)  # Parameter value for the curve
            derivatives = curve.derivatives(u=u, order=2)  # Get first and second derivatives at point
            curvature = self.calculate_curvature(derivatives[1], derivatives[2])  # Calculate curvature
            total_curvature += curvature

        average_curvature = total_curvature / num_points
        return average_curvature
    
    def calculate_nurbs_path_distance(self, curve):

        points = curve.evalpts
        total_distance = sum(np.linalg.norm(np.array(points[i]) - np.array(points[i+1])) for i in range(len(points) - 1))
        return total_distance
    
    def check_constraints(self, curve, num_samples=100):
        last_point = None
        for i in range(num_samples):
            t = i / float(num_samples - 1)
            point = curve.evaluate_single(t)

            # Check if the point is in an obstacle or out of bounds
            if self.env.is_point_in_obstacle(point) or not self.env.is_point_within_bounds(point):
                if self.env.is_point_in_obstacle(point):
                    print("in obstacle")
                if not self.env.is_point_within_bounds(point):
                    print("out of bounds")
                return False

            # Check for line segment intersection with obstacles, if there's a last point
            if last_point is not None and self.env.is_line_intersecting_obstacles(last_point, point):
                print("line intersection")
                return False

            last_point = point

        return True  # All constraints satisfied

    def evaluate_individual(self, individual):
        # Construct a Curve object from the individual
        curve = NURBS.Curve()
        curve.degree = 3
        curve.ctrlpts = individual[0]  # Control points
        curve.weights = individual[1]  # Weights
        curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))

        # Check if the individual meets the constraints
        if not self.check_constraints(curve):
            # Assign a high penalty for failing constraints
            # print("Failed constraints")
            return float('inf'), float('inf')  # Represents a very bad fitness value for both objectives

        # calculate its fitness values
        distance = self.calculate_nurbs_path_distance(curve)
        
        smoothness = self.calculate_smoothness(curve)
        inverted_smoothness = 1 / smoothness if smoothness != 0 else float('inf')  # Avoid division by zero

        return distance, inverted_smoothness

    def run_evolution(self, initial_population, max_generations, no_improve_generations):
        # Use the provided initial population
        population = initial_population

        best_fitness = None
        generations_without_improvement = 0

        for gen in range(max_generations):
            print(f"Evaluating population for generation {gen + 1}/{max_generations}...")

            # Evaluate and assign fitness to each individual
            fitnesses = [self.evaluate_individual(ind) for ind in population]
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Breeding and Mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    original_child1 = copy.deepcopy(child1)
                    original_child2 = copy.deepcopy(child2)
                    self.toolbox.mate(child1, child2)

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    original_mutant = copy.deepcopy(mutant)
                    self.toolbox.mutate(mutant)

            # Evaluate the offspring with an invalid fitness
            for ind in [ind for ind in offspring if not ind.fitness.valid]:
                ind.fitness.values = self.evaluate_individual(ind)

            # Convert offspring back to NURBS paths for plotting
            nurbs_offspring = []
            for ind in offspring:
                curve = NURBS.Curve()
                curve.degree = 3
                curve.ctrlpts = ind[0]  # Assuming first element is control points
                curve.weights = ind[1]  # Assuming second element is weights
                curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))
                nurbs_offspring.append(curve)

            # Plot paths at certain generations
            self.env.plot_paths_on_environment(nurbs_offspring, is_nurbs=True)

            # Replace the old population with the new offspring
            population[:] = offspring

            # Logging and checking improvements
            current_best_fitness = min(ind.fitness.values for ind in population)
            print(f"Generation {gen + 1}: Best Fitness - {current_best_fitness}")
            if best_fitness is None or current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= no_improve_generations:
                print(f"No improvement after {no_improve_generations} generations")
                break

        return population




def main():
    
    

    """
    print("Initializing environment...")
    env = PathPlanningEnvironment(100, 100, 1)  

    print("Creating random obstacles...")
    # fixed_obstacles = [(5, 5, 1), (15, 15, 1)]  # Fixed obstacles for consistent output
    # env.set_fixed_obstacles(fixed_obstacles)
    num_obstacles_range = (5, 15)
    obstacle_radius_range = (1, 5)
    env.create_random_obstacles(num_obstacles_range, obstacle_radius_range)

    # Visualize the environment
    # env.plot_environment()

    # Generate and visualize the graph
    print("Converting to graph...")
    G = env.convert_to_graph()
    # env.visualize_graph(G)

    print("Finding shortest paths...")
    # Define start and goal points
    start = (0, 0)
    goal = (99, 99)

    # Find multiple shortest paths using Yen's K Shortest Paths Algorithm
    K = 10  # Number of paths to find
    paths = env.yen_k_shortest_paths(start, goal, K)

    # # Convert paths to NURBS curves
    print("Converting shortest paths to NURBS curves...")
    yen_nurbs_paths = [env.convert_path_to_nurbs(path) for path in paths]


    # Generate and convert random paths to NURBS curves
    print("Finding random paths...")
    num_random_paths = 5
    num_control_points = 10
    noise_variance = 10
    random_paths = env.generate_random_paths(start, goal, num_random_paths, num_control_points, noise_variance)

    print("Converting random paths to NURBS curves...")
    random_nurbs_paths = [env.convert_path_to_nurbs(path, is_grid_coords=False) for path in random_paths]

    combined_nurbs_paths = yen_nurbs_paths + random_nurbs_paths

    # Save the environment to a file
    with open('path_planning_environment.pkl', 'wb') as file:
        pickle.dump(env, file)

    print("Environment saved to 'path_planning_environment.pkl'")

    with open('nurbs_paths.pkl', 'wb') as file:
        pickle.dump(combined_nurbs_paths, file)

    print("NURBS paths saved to 'nurbs_paths.pkl'")

    # # Plot all paths on the environment
    # if paths:
    #     # Plot the NURBS paths
    #     env.plot_paths_on_environment(combined_nurbs_paths, is_nurbs=True)
    #     # Plot the grid paths
    #     env.plot_paths_on_environment(paths)

    # else:
    #     print("No paths found")

    """

    # Saved initial population for debugging
    if os.path.exists('path_planning_environment.pkl'):
        print("Loading environment from 'path_planning_environment.pkl'...")
        with open('path_planning_environment.pkl', 'rb') as file:
            env = pickle.load(file)

        # Also load the combined NURBS paths if they exist
        if os.path.exists('nurbs_paths.pkl'):
            with open('nurbs_paths.pkl', 'rb') as file:
                combined_nurbs_paths = pickle.load(file)

    env.plot_paths_on_environment(combined_nurbs_paths, is_nurbs=True)

    # Initialize the evolutionary path planner
    print("Initializing evolutionary path planner...")
    crossover_prob = 0.7  # Example crossover probability
    mutation_prob = 0.2   # Example mutation probability
    epp = EvolutionaryPathPlanner(env, crossover_prob, mutation_prob)

    initial_population = []
    for nurbs_curve in combined_nurbs_paths:
        individual = (nurbs_curve.ctrlpts, nurbs_curve.weights)
        initial_population.append(creator.Individual(individual))

    
    for idx, nurbs_curve in enumerate(combined_nurbs_paths):
        constraints_passed = epp.check_constraints(nurbs_curve)
        print(f"Path {idx + 1}: Constraints {'passed' if constraints_passed else 'failed'}")

    # Select two individuals for breeding and one for mutation
    parent1 = initial_population[0]
    parent2 = initial_population[14]
    individual_for_mutation = initial_population[2]

    # Apply breeding
    child_cp, child_weights = epp.breeding(parent1, parent2)
    child_curve = NURBS.Curve()
    child_curve.degree = 3
    child_curve.ctrlpts = child_cp
    child_curve.weights = child_weights
    child_curve.knotvector = utils.generate_knot_vector(child_curve.degree, len(child_curve.ctrlpts))

    # Apply mutation
    mutated_cp, mutated_weights = epp.mutation(individual_for_mutation, len(individual_for_mutation[0]))
    mutated_curve = NURBS.Curve()
    mutated_curve.degree = 3
    mutated_curve.ctrlpts = mutated_cp
    mutated_curve.weights = mutated_weights
    mutated_curve.knotvector = utils.generate_knot_vector(mutated_curve.degree, len(mutated_curve.ctrlpts))

    # Convert parent Individuals back to NURBS curves for plotting
    def convert_individual_to_curve(individual):
        curve = NURBS.Curve()
        curve.degree = 3
        curve.ctrlpts = individual[0]
        curve.weights = individual[1]
        curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        return curve

    parent1_curve = convert_individual_to_curve(parent1)
    parent2_curve = convert_individual_to_curve(parent2)
    individual_for_mutation_curve = convert_individual_to_curve(individual_for_mutation)

    # Plot original parents and child
    env.plot_paths_on_environment([parent1_curve, parent2_curve, child_curve], is_nurbs=True)

    # Plot original individual and mutated version
    env.plot_paths_on_environment([individual_for_mutation_curve, mutated_curve], is_nurbs=True)
    
    # Run the evolutionary algorithm with the initial paths
    print("Running evolutionary algorithm...")
    final_population = epp.run_evolution(initial_population, max_generations=100, no_improve_generations=10)

    print("Evolution complete. Processing final population...")




if __name__ == '__main__':
    multiprocessing_setup()  # Set up multiprocessing
    main()  # Run the main function