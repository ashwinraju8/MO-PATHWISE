"""

import random
from deap import base, creator, tools, benchmarks

# Define the problem as a Minimization problem for multi-objective optimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
BOUND_LOW, BOUND_UP = 0.0, 1.0
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

# Adjusted parameters
CROSSOVER_RATE = 0.9  # Increased crossover rate
MUTATION_RATE = 0.2   # Adjusted mutation rate
NO_IMPROVE_GENERATIONS = 20  # Increased threshold for early termination


def run_evolution(self, initial_population, max_generations, no_improve_generations):
    population = initial_population
    best_fitness = None
    generations_without_improvement = 0

    for gen in range(max_generations):
        print(f"Evaluating population for generation {gen + 1}/{max_generations}...")

        # Evaluate and assign fitness to each individual
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Breeding and Mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring with an invalid fitness
        for ind in [ind for ind in offspring if not ind.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replace the old population
        population[:] = offspring

        # Logging and checking improvements
        current_best_fitness = min(ind.fitness.values for ind in population)
        print(f"Generation {gen + 1}: Best Fitness - {current_best_fitness}")

        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= NO_IMPROVE_GENERATIONS:
            print(f"No significant improvement after {NO_IMPROVE_GENERATIONS} generations, stopping...")
            break

    return population

# Initialize population and run the evolution
initial_population = toolbox.population(n=50)
final_population = run_evolution(toolbox, initial_population, 100, 10)





   This file is part of DEAP.

   DEAP is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation, either version 3 of
   the License, or (at your option) any later version.

   DEAP is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""

"""

import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 250
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook

if __name__ == "__main__":
    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))

    pop, stats = main()
    # pop.sort(key=lambda x: x.fitness.values)

    # print(stats)
    # print("Convergence: ", convergence(pop, optimal_front))
    # print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))

    # import matplotlib.pyplot as plt
    # import numpy

    # front = numpy.array([ind.fitness.values for ind in pop])
    # optimal_front = numpy.array(optimal_front)
    # plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    # plt.scatter(front[:,0], front[:,1], c="b")
    # plt.axis("tight")
    # plt.show()

"""

"""

import random
from deap import base, creator, tools, benchmarks

# Define problem for multi-objective optimization
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Problem setup
toolbox = base.Toolbox()
BOUND_LOW, BOUND_UP = 0.0, 1.0
NDIM = 30

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def run_evolution(toolbox, pop_size=50, gen_count=100, no_improve_limit=20):
    population = toolbox.population(n=pop_size)
    best_fitness = None
    generations_without_improvement = 0

    for gen in range(gen_count):
        # Evaluate the entire population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        # Select and clone the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.1:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the offspring with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Replace the old population with the new offspring
        population[:] = offspring

        # Logging and checking improvements
        current_best_fitness = min(ind.fitness.values for ind in population)
        print(f"Generation {gen + 1}/{gen_count}, Best Fitness: {current_best_fitness}")

        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            generations_without_improvement = 0
            print(f"Improvement in generation {gen + 1}")
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= no_improve_limit:
            print(f"No significant improvement after {no_improve_limit} generations, stopping...")
            break

    return population

# Run the evolution
final_population = run_evolution(toolbox)

"""

import array
import random
import json

import numpy

from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap.benchmarks.tools import hypervolume

# Problem configuration
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
BOUND_LOW, BOUND_UP = 0.0, 1.0
NDIM = 30

def uniform(low, up, size=None):
    return [random.uniform(a, b) for a, b in zip([low]*size, [up]*size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main():
    random.seed(64)
    NGEN = 10000  # Number of generations
    MU = 100    # Population size
    CXPB = 0.9  # Crossover probability
    no_improve_limit = 500  # Stopping condition
    no_improve_gen = 0  # Counter for generations without improvement
    best_fitness = None

    pop = toolbox.population(n=MU)

    # Evaluate the initial population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Assign crowding distance
    pop = toolbox.select(pop, len(pop))

    # Begin the generational process
    for gen in range(1, NGEN):
        # Select and clone the next generation individuals
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select and assign crowding distance to the next generation population
        pop[:] = toolbox.select(pop + offspring, MU)

        # Check for improvement
        current_best_fitness = min(ind.fitness.values[0] for ind in pop)
        if best_fitness is None or current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            no_improve_gen = 0
        else:
            no_improve_gen += 1

        # Check stopping condition
        if no_improve_gen >= no_improve_limit:
            print(f"No improvement for {no_improve_limit} generations. Stopping at generation {gen}.")
            break

        # Optional: Print best fitness in the population
        print(f"Gen: {gen}, Best Fitness: {current_best_fitness}")

    print("Final population hypervolume:", hypervolume(pop, [11.0, 11.0]))
    return pop

if __name__ == "__main__":
    final_population = main()
