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
from deap.tools.emo import sortNondominated, assignCrowdingDist
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
        # Assign a default weight between 1 and 1.5 to all edges
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
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", self.breeding)
        self.toolbox.register("mutate", self.mutation, n_control_points=50)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self.evaluate_individual)
  
    def breeding(self, ind1, ind2):
        # Extract control points, weights, and sigma values from each parent
        cp1, w1, sigmaxy1, sigmaw1 = ind1
        cp2, w2, sigmaxy2, sigmaw2 = ind2

        # Calculate the average of sigmaxy1 and sigmaxy2
        child_sigma_xy = (
            [(x1 + x2) / 2 for x1, x2 in zip(sigmaxy1[0], sigmaxy2[0])],
            [(y1 + y2) / 2 for y1, y2 in zip(sigmaxy1[1], sigmaxy2[1])]
        )

        child_sigma_w = [(w1 + w2) / 2 for w1, w2 in zip(sigmaw1, sigmaw2)]

        # Update the individuals in-place
        ind1[:] = [cp1, w1, child_sigma_xy, child_sigma_w]

        # Create a copy of each list in the child_sigma_xy tuple
        child_sigma_xy_copy = (child_sigma_xy[0][:], child_sigma_xy[1][:])

        # Use the copied tuple for ind2
        ind2[:] = [cp2, w2, child_sigma_xy_copy, child_sigma_w[:]]
            
    def mutation(self, individual, n_control_points):
        # Unpack individual's attributes
        ctrlpts, weights, sigmaxy, sigmaw = individual
        ctrlpts_x, ctrlpts_y = zip(*ctrlpts)  # Unpack control points into x and y lists
        sigmaxy_x, sigmaxy_y = sigmaxy  # Unpack sigmaxy into x and y lists

        # Calculate mutation parameters
        D = 3 * n_control_points - 4  # Dimensionality
        l = 0.5  # Hyperparameter
        tau0 = l / np.sqrt(2 * D)  # Global step size scaling
        tau = l / np.sqrt(2 * np.sqrt(D))  # Local step size scaling

        # Global step size mutation
        xi0 = np.random.normal()  # Random number from standard normal distribution
        global_step = np.exp(tau0 * xi0)  # Global step size

        # Mutate step sizes for x and y coordinates of control points
        mutated_sigmaxy_x = []
        mutated_sigmaxy_y = []
        for sigmax, sigmay in zip(sigmaxy_x, sigmaxy_y):
            xi_x = np.random.normal()  # Random number for x-coordinate
            xi_y = np.random.normal()  # Random number for y-coordinate
            mutated_sigmaxy_x.append(sigmax * np.exp(tau * xi_x))
            mutated_sigmaxy_y.append(sigmay * np.exp(tau * xi_y))

        # Mutate step sizes for weights
        mutated_sigmaw = [sigma * np.exp(tau * np.random.normal()) for sigma in sigmaw]

        # Apply mutation to control points using mutated step sizes
        mutated_ctrlpts_x = [ctrlpts_x[0]] + [ctrlpt + global_step * sigma * np.random.normal() for ctrlpt, sigma in zip(ctrlpts_x[1:-1], mutated_sigmaxy_x)] + [ctrlpts_x[-1]]
        mutated_ctrlpts_y = [ctrlpts_y[0]] + [ctrlpt + global_step * sigma * np.random.normal() for ctrlpt, sigma in zip(ctrlpts_y[1:-1], mutated_sigmaxy_y)] + [ctrlpts_y[-1]]

        # Apply mutation to weights
        mutated_weights = [weight + global_step * sigma * np.random.normal() for weight, sigma in zip(weights, mutated_sigmaw)]

        # Update the individual with mutated values
        mutated_ctrlpts = list(zip(mutated_ctrlpts_x, mutated_ctrlpts_y))
        individual[:] = [mutated_ctrlpts, mutated_weights, (mutated_sigmaxy_x, mutated_sigmaxy_y), mutated_sigmaw]

    def distance_to_nearest_obstacle(self, curve, num_samples=100):
        min_distance = float('inf')

        for i in range(num_samples):
            t = i / float(num_samples - 1)
            point = curve.evaluate_single(t)
            
            # Calculate distance to each obstacle for this point
            for ox, oy, radius in self.env.obstacles:
                distance = np.sqrt((point[0] - ox) ** 2 + (point[1] - oy) ** 2) - radius
                min_distance = min(min_distance, distance)

        return max(min_distance, 0)  # Ensure distance is non-negative

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

        safety_distance = self.distance_to_nearest_obstacle(curve)
        inverted_safety_distance = 1 / safety_distance if safety_distance != 0 else float('inf')  # Avoid division by zero

        # return distance, inverted_smoothness
        return distance, inverted_safety_distance
    
    def run_evolution(self, initial_population, max_generations, no_improve_limit):
        # Ensure population size is divisible by 4 for selTournamentDCD
        while len(initial_population) % 4 != 0:
            initial_population.append(copy.deepcopy(initial_population[-1]))

        population = initial_population

        # Initialize variables for tracking Pareto front
        all_pareto_fronts = []
        no_improve_gen = 0

        # Function to calculate diversity
        def calculate_diversity(population):
            unique_individuals = set(map(str, population))
            return len(unique_individuals)
        
        def print_individuals_change(before, after, operation):
            print(f"Changes due to {operation}:")
            for i, (ind_before, ind_after) in enumerate(zip(before, after)):
                # Unpack the control points, weights, and sigma values
                ctrlpts_before, weights_before, sigmaxy_before, sigmaw_before = ind_before
                ctrlpts_after, weights_after, sigmaxy_after, sigmaw_after = ind_after

                # Check for changes in control points, weights, or sigma values
                if (ctrlpts_before != ctrlpts_after or 
                    weights_before != weights_after or 
                    sigmaxy_before != sigmaxy_after or 
                    sigmaw_before != sigmaw_after):
                    print(f"  Individual {i} changed.")

        # Evaluate the initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        population = self.toolbox.select(population, len(population))

        for gen in range(max_generations):
            # Diversity and population size tracking
            diversity = calculate_diversity(population)
            print(f"Generation {gen + 1}: Population size: {len(population)}, Diversity: {diversity}")

            # Vary the population
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            print_individuals_change(population, offspring, "selTournamentDCD")

            # Track changes before and after breeding and mutation
            breeding_before = copy.deepcopy(offspring)
            mutation_before = copy.deepcopy(offspring)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            print_individuals_change(breeding_before, offspring, "Breeding")

            for mutant in offspring:
                print("Mutating")
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

            print_individuals_change(mutation_before, offspring, "Mutation")

            # Evaluate individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_count = len(invalid_ind)
            print(f"Generation {gen + 1}: Invalid individuals before evaluation: {invalid_count}")

            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            invalid_count = sum(1 for ind in offspring if not ind.fitness.valid)
            print(f"Generation {gen + 1}: Invalid individuals after evaluation: {invalid_count}")

            # # Convert offspring back to NURBS paths for plotting
            # nurbs_offspring = []
            # for ind in offspring:
            #     curve = NURBS.Curve()
            #     curve.degree = 3
            #     curve.ctrlpts = ind[0]  # Assuming first element is control points
            #     curve.weights = ind[1]  # Assuming second element is weights
            #     curve.knotvector = utils.generate_knot_vector(curve.degree, len(curve.ctrlpts))
            #     nurbs_offspring.append(curve)

            # # Plot paths at certain generations
            # self.env.plot_paths_on_environment(nurbs_offspring, is_nurbs=True)

            # Check selection process
            selection_before = copy.deepcopy(population + offspring)
            population[:] = self.toolbox.select(population + offspring, len(population))
            print_individuals_change(selection_before, population, "Selection")

            # Get the current Pareto front
            current_pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

            # Save current Pareto front
            all_pareto_fronts.append(current_pareto_front)

            # Check for improvement compared to the previous Pareto front (if there is one)
            if gen > 0 and any(ind.fitness.dominates(prev_ind.fitness) for ind in current_pareto_front for prev_ind in all_pareto_fronts[-2]):
                no_improve_gen = 0
            else:
                no_improve_gen += 1

            # Early stopping condition if no improvement in Pareto front for a set number of generations
            if no_improve_gen >= no_improve_limit:
                print(f"Stopped early due to no improvement over {no_improve_limit} generations.")
                break

            print(f"Generation {gen + 1}, Pareto Front Size: {len(current_pareto_front)}")

        # Plotting Pareto fronts at the end of the evolution
        total_generations = len(all_pareto_fronts)
        selected_generations = np.linspace(0, total_generations - 1, 5, dtype=int)

        # Plotting selected Pareto fronts
        for gen_index in selected_generations:
            pareto_front = all_pareto_fronts[gen_index]
            # Extract fitness values for each individual in the Pareto front
            front = np.array([ind.fitness.values for ind in pareto_front])
            # Plot
            plt.scatter(front[:, 0], front[:, 1], label=f"Gen {gen_index + 1}")

        plt.title("Selected Pareto Fronts Over Generations")
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.legend()
        plt.show()

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
        # Get control points and weights
        ctrlpts = nurbs_curve.ctrlpts
        weights = nurbs_curve.weights
        
        # Initialize sigma values
        sigmas_x = [1.0] * len(ctrlpts)
        sigmas_y = [1.0] * len(ctrlpts)
        sigmas_xy = (sigmas_x,sigmas_y)
        sigmas_w = [0.0] * len(weights)
        
        # Combine control points, weights, and sigmas
        individual = (ctrlpts, weights, sigmas_xy, sigmas_w)
        
        # Append the individual to the initial population
        initial_population.append(creator.Individual(individual))

    # Testing functions:
    """
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
    """

    # Run the evolutionary algorithm with the initial paths
    print("Running evolutionary algorithm...")
    final_population = epp.run_evolution(initial_population, max_generations=100, no_improve_limit=20)

    print("Evolution complete. Processing final population...")
    



if __name__ == '__main__':
    multiprocessing_setup()  # Set up multiprocessing
    main()  # Run the main function