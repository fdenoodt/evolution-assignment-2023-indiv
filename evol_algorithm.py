# import random

from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from diversity import Island, FitnessSharing
from placket_luce import PlackettLuce

import numpy as np

from selection import Selection
from variation import Variation


class EvolAlgorithm(AbstractAlgorithm):
    def __init__(self, benchmark, popul_size=1000, offspring_size_multiplier=2, k=3, mutation_rate=0.05,
                 migrate_after_epochs=25):
        self.benchmark = benchmark

        self.popul_size = popul_size
        self.offspring_size = offspring_size_multiplier * popul_size
        self.k = k  # Tournament selection
        self.mutation_rate = mutation_rate
        self.keep_running_until_timeup = True
        self.migrate_after_epochs = migrate_after_epochs

        super().__init__()

    def optimize(self, numIters, keep_running_until_timeup, reporter_name, *args):
        n = self.benchmark.permutation_size()

        # since zero is implicit, we need to the edge from 0 to first node
        f = lambda population: self.benchmark.compute_fitness(population) + self.benchmark.matrix[0, population[:, 0]]
        maximize = self.benchmark.maximise
        keep_running_until_timeup = keep_running_until_timeup
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.benchmark)

        selection = lambda population, fitnesses: (
            Selection.selection(population, self.k, self.offspring_size, fitnesses))
        elimination = lambda popul, fitnesses: Selection.elimination(popul, fitnesses, self.k, self.popul_size)
        mutation = lambda offspring: Variation.mutation(offspring, self.mutation_rate)
        crossover = lambda selected: Variation.crossover(selected)

        islands = [Island(idx, f, self.popul_size, n) for idx in range(5)]

        ctr = 0
        done = False
        while not (done):
            # run for a few epochs
            # Time to run for a few epochs
            done = Island.run_epochs(self.migrate_after_epochs, islands,
                                     selection, elimination, mutation, crossover,
                                     score_tracker, ctr)

            # migrate
            Island.migrate(islands, self.popul_size)

            ctr += 1

        return score_tracker.all_time_best_fitness


if __name__ == "__main__":
    n = 4

    # parent = np.array([1, 3, 0, 2]) #np.random.permutation(n)
    # parent = np.array([0, 3, 1, 2])  # np.random.permutation(n)

    print("*" * 20)
    parent = np.array([2, 1, 3])  # 0, 2, 1, 3 but 0 is implicit
    e = EvolAlgorithm(None)
    parent_cyclic = Variation.edge_table(parent, n)  # 2 3 1 0
    print(parent)
    print(parent_cyclic)

    print("*" * 20)
    print("Testing population initialization")
    popul = e.initialize_population(10, n)
    print(popul.shape)
    print(popul)

    print("*" * 20)
    print("Testing crossover")
    # selected = np.array([popul[0], popul[0]])  # 2x same individual --> so crossover should be same
    # selected = np.array(([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]), dtype=int)
    selected = np.array(([1, 2], [1, 2]), dtype=int)
    print(selected)
    offspring = Variation.crossover(selected)
    print(offspring)

    print("*" * 20)
    print("Testing distance")

    print(FitnessSharing.distance(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])))
    print(FitnessSharing.distance(np.array([1, 2, 3, 4]), np.array([1, 2, 4, 3])))

    print("*" * 20)
    print("Testing fitness sharing")
    fitnesses = np.array([100, 100, 100, 100, 100], dtype=np.float64)
    population = np.array([[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], [1, 4, 2, 3]])

    # compare time between fitness sharing and fitness sharing fast
    import time

    n = 100
    fitnesses = np.random.rand(n)
    population = np.random.rand(n, n)
    e = EvolAlgorithm(None)

    time1 = time.time()
    for i in range(10):
        e.fitness_sharing(fitnesses, population)

    time2 = time.time()
    print("time1:", time2 - time1)

    time1 = time.time()
    for i in range(10):
        e.fitness_sharing_slow(fitnesses, population)

    time2 = time.time()
    print("time2:", time2 - time1)
