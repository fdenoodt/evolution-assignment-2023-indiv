# import random

from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from diversity import Island, FitnessSharing
from local_search import LocalSearch
from placket_luce import PlackettLuce

import numpy as np

from selection import Selection
from variation import Variation


class EvolAlgorithm(AbstractAlgorithm):
    def __init__(self, benchmark, hyperparams, filename=None):
        self.benchmark = benchmark
        assert (benchmark.normalizing_constant == 1), \
            "Normalizing for EvolAlgorithm gives no benefits and should be disabled"

        self.popul_size = hyperparams.popul_size
        self.offspring_size = hyperparams.offspring_size_multiplier * self.popul_size
        self.k = hyperparams.k
        self.mutation_rate = hyperparams.mutation_rate
        self.keep_running_until_timeup = hyperparams.keep_running_until_timeup
        self.migrate_after_epochs = hyperparams.migrate_after_epochs
        self.nb_islands = hyperparams.nb_islands
        self.migration_percentage = hyperparams.migration_percentage
        self.merge_after_percent_time_left = hyperparams.merge_after_percent_time_left
        self.fitness_subset_percentage = hyperparams.fitness_sharing_subset_percentage
        self.alpha_sharing = hyperparams.alpha

        self.local_search = hyperparams.local_search[0]
        self.local_search_param = hyperparams.local_search[1]

        self.filename = filename  # used for saving the results

        super().__init__()

    def optimize(self, numIters, reporter_name, *args):
        n = self.benchmark.permutation_size()

        # since zero is implicit, we need to the edge from 0 to first node
        f = lambda population: self.benchmark.compute_fitness(population) + self.benchmark.matrix[0, population[:, 0]]
        maximize = self.benchmark.maximise
        keep_running_until_timeup = self.keep_running_until_timeup
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.benchmark,
                                     self.filename)

        selection = lambda population, fitnesses: (
            Selection.selection(population, self.k, self.offspring_size, fitnesses))
        elimination = lambda popul, fitnesses: Selection.elimination(popul, fitnesses, self.k, self.popul_size)

        fitness_sharing = lambda fitnesses, population: FitnessSharing.fitness_sharing(
            fitnesses, population,
            self.fitness_subset_percentage,
            self.alpha_sharing)

        # local search
        assert (self.local_search in [None, "2-opt", "insert_random_node"]), "Invalid local search"
        if self.local_search == "2-opt":
            # assert whole number >= 1
            assert (self.local_search_param >= 1) and (self.local_search_param % 1 == 0), \
                "Invalid local search param for 2-opt"

            jump_size = self.local_search_param
            local_search = lambda offspring: LocalSearch.two_opt(offspring,
                                                                 score_tracker.benchmark.matrix,
                                                                 jump_size=jump_size)
        elif self.local_search == "insert_random_node":
            # assert 0 <= param <= 1
            assert (0 <= self.local_search_param <= 1), "Invalid local search param for insert_random_node"

            nb_nodes_to_insert_percent = self.local_search_param
            local_search = lambda offspring: LocalSearch.insert_random_node(offspring,
                                                                            score_tracker.benchmark.matrix,
                                                                            nb_nodes_to_insert_percent=nb_nodes_to_insert_percent)
        else:
            local_search = lambda offspring: offspring  # do nothing

        # Ensure that we have a pair of every mutation and crossover combination
        mutation_functions = [
            lambda offspring: Variation.swap_mutation(offspring, self.mutation_rate),
            lambda offspring: Variation.inversion_mutation(offspring, self.mutation_rate),
            lambda offspring: Variation.scramble_mutation(offspring, self.mutation_rate),
        ]

        # didn't result in good performance
        # crossover_functions = [
        #     lambda selected: Variation.crossover(selected),
        #     lambda selected: Variation.order_crossover(selected),
        # ]

        crossover_functions = [
            # with probability 0.2 use edge crossover, otherwise use order crossover
            lambda selected: Variation.edge_crossover(selected) \
                if np.random.random() < 0.2 \
                else Variation.order_crossover(selected),
        ]

        # set names of the lambda functions for easy printing
        mutation_functions[0].__name__ = "swap_mut"
        mutation_functions[1].__name__ = "inversion_mut"
        mutation_functions[2].__name__ = "scramble_mut"

        # crossover_functions[0].__name__ = "edge_cross"
        # crossover_functions[1].__name__ = "order_cross"
        crossover_functions[0].__name__ = "edge/order_cross"

        # create a list of all possible combinations
        functions = np.array([
            (mutation, crossover) for mutation in mutation_functions for crossover in crossover_functions
        ])

        # shuffle the list
        np.random.shuffle(functions)

        islands = [Island(idx, f, self.popul_size, n,
                          mutation=functions[idx % len(functions)][0],
                          crossover=functions[idx % len(functions)][1])
                   for idx in range(self.nb_islands)]

        ctr = 0
        done = False
        has_merged = False
        while not (done):
            done, time_left = Island.run_epochs(self.migrate_after_epochs, islands,
                                                selection, elimination, fitness_sharing,
                                                local_search,
                                                score_tracker, ctr)

            # migrate
            Island.migrate(islands, self.popul_size, percentage=self.migration_percentage)

            # if half time left, merge islands
            if time_left < self.merge_after_percent_time_left * score_tracker.utility.reporter.allowedTime and not has_merged:
                # Merge all islands into one and run for the remaining time, w/ edge crossover as it converges faster
                print("*" * 20)
                print("Merging islands")
                print()
                crossover = Variation.edge_crossover
                mutation = lambda offspring: Variation.scramble_mutation(offspring, self.mutation_rate)
                mutation.__name__ = "scramble_mut"
                island = Island.merge_islands(islands, crossover, mutation)
                islands = [island]
                has_merged = True

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
