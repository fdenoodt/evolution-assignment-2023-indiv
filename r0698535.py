import reporter as Reporter
from utility import Utility
from abc import ABC, abstractmethod
import sys
import time
from python_tsp.exact import solve_tsp_dynamic_programming  # pip install python-tsp
from python_tsp.heuristics import solve_tsp_simulated_annealing
import matplotlib.pyplot as plt

import numpy as np


class AbstractBenchmark(ABC):
    def __init__(self, matrix, normalize, maximise):
        self.normalizing_constant = 1
        self.matrix = matrix
        self.maximise = maximise
        if normalize:
            self.matrix, self.normalizing_constant = AbstractBenchmark.normalize_matrix(self.matrix)

    def permutation_size(self):
        return self.matrix.shape[0]

    @staticmethod
    def normalize_matrix(matrix):
        # normalize distance matrix to be between 0 and 1
        # it makes the w's smaller and thus less likely to overflow
        constant = np.max(matrix)
        distanceMatrix = matrix / constant
        return distanceMatrix, constant

    def unnormalize_fitnesses(self, fitnesses):
        fitnesses = fitnesses if self.normalizing_constant == 1 else fitnesses * self.normalizing_constant
        return fitnesses

    @abstractmethod
    def compute_fitness(self, population):
        pass


class Benchmark(AbstractBenchmark):
    def __init__(self, filename, normalize, maximise, replace_inf_with_large_val=True):
        # Read distance matrix from file.
        file = open(filename)
        _matrix = np.loadtxt(file, delimiter=",")
        file.close()

        if replace_inf_with_large_val:
            _matrix = Benchmark.replace_inf_with_large_val(_matrix)

        # TODO: remove
        # _matrix = _matrix[:10, :10]

        super().__init__(_matrix, normalize, maximise)

    @staticmethod
    def replace_inf_with_large_val(distanceMatrix):
        # replace inf with largest non inf value * max number of cities
        # just max is not enough, needs to make sure that worst possible path is still better than a single inf
        largest_value = np.max(distanceMatrix[distanceMatrix != np.inf]) * len(distanceMatrix)
        distanceMatrix = np.where(distanceMatrix == np.inf,
                                  largest_value, distanceMatrix)
        # faster for the start, finds existing solutions quicker but in long run not that much impact

        print(f"largest non inf val: {largest_value:_.4f}")
        return distanceMatrix

    def compute_fitness_slow(self, population):  # slow, but easy to understand
        # shape: (populationSize, numCities)
        # eg population: [[1,2,3,4,5],[1,2,3,4,5], ... ]

        fitnesses = []
        for i in range(len(population)):
            individual = population[i]
            fitness = 0
            for j in range(len(individual)):
                city = individual[j]
                nextCity = individual[(j + 1) % len(individual)]
                fitness += self.matrix[int(city)][int(nextCity)]

            fitnesses.append(fitness)
        return np.array(fitnesses)

    def compute_fitness_explicit(self, population):  # faster, generated with copilot but we understand it!
        distanceMatrix = self.matrix

        # assert population doesn't contain cities that are floats (sanity check, can be removed later)
        assert np.all(np.equal(np.mod(population, 1), 0))

        # the faster way
        fitnesses = np.array([
            np.sum([distanceMatrix[int(city)][int(nextCity)] \
                    for city, nextCity in zip(individual, np.roll(individual, -1))])
            for individual in population])

        # returns: (populationSize, 1)
        # eg: [100,200, ... ]
        return fitnesses

    def compute_fitness(self, population):
        # https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
        # In fact, exactly the same as the explicit version
        return np.array([self.matrix[route, np.roll(route, -1)].sum() for route in population])

    def dp_solve(self):
        distance_matrix = self.matrix  # [:10, :10]
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)

        return permutation, distance

    def meta_solve(self):
        distance_matrix = self.matrix  # [:10, :10]
        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

        return permutation, distance


class AbstractAlgorithm(ABC):

    @abstractmethod
    # variable number of arguments
    def optimize(self, *args):
        pass


class PlackettLuceAlgorithm(AbstractAlgorithm):
    def __init__(self, lr, nb_samples_lambda, U, benchmark, pdf, keep_running_until_timeup, filename=None):
        assert not (benchmark.normalizing_constant == 1), \
            "Normalizing for PlackettLuceAlgorithm is required to prevent overflow, so it should be enabled"

        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda
        self.keep_running_until_timeup = keep_running_until_timeup

        self.pl = PlackettLuce(U, benchmark)
        self.pdf = pdf

        self.filename = filename  # used for saving the results (other than ./r01235..)

        super().__init__()

    def optimize(self, numIters, reporter_name, max_duration=None, *args):
        print("************** STARTING")
        print(reporter_name)

        # *args can be used as follows:
        # for arg in args:
        #     print(arg)

        # or via indexing:
        # print(args[0])

        # or via unpacking:
        # a, b, c = args

        n = self.pl.benchmark.permutation_size()
        f = self.pl.benchmark.compute_fitness
        keep_running_until_timeup = self.keep_running_until_timeup

        pdf = self.pdf
        maximize = self.pl.benchmark.maximise
        keep_running_until_timeup = keep_running_until_timeup

        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.pl.benchmark,
                                     second_filename=self.filename, max_duration=max_duration)

        self.optimize_plackett_luce(f, self.lr, self.nb_samples_lambda, n, pdf, maximize, score_tracker)

    def optimize_plackett_luce(self, fitness_func, lr, nb_samples_lambda, n, pdf, maximize, score_tracker):
        ctr = 0
        while True:
            # Sample sigma_i from Plackett luce
            sigmas = pdf.sample_permutations(nb_samples_lambda)
            fitnesses = fitness_func(sigmas)

            delta_w_log_ps = pdf.calc_gradients(sigmas)
            best_fitness, mean_fitness, sigma_best = score_tracker.update_scores(
                fitnesses, sigmas, ctr,
                fitnesses_shared=None,  # only used in EvolAlgorithm
                pdf=pdf,
                print_w=True)
            delta_w_log_F = PlackettLuce.calc_w_log_F(self.pl.U, fitnesses, delta_w_log_ps, nb_samples_lambda)
            pdf.update_w_log(delta_w_log_F, lr, maximize)

            ctr += 1
            is_done, time_left = score_tracker.utility.is_done_and_report(ctr, mean_fitness, best_fitness, sigma_best,
                                                                          write_to_file=True)
            if is_done:
                break

        return score_tracker.all_time_best_fitness


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

    def optimize(self, numIters, reporter_name, max_duration=None, *args):
        n = self.benchmark.permutation_size()

        # since zero is implicit, we need to the edge from 0 to first node
        f = lambda population: self.benchmark.compute_fitness(population) + self.benchmark.matrix[0, population[:, 0]]
        maximize = self.benchmark.maximise
        keep_running_until_timeup = self.keep_running_until_timeup
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.benchmark,
                                     second_filename=self.filename, max_duration=max_duration)

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
        best_fitness = np.inf
        while not (done):
            done, time_left, best_fitness = Island.run_epochs(self.migrate_after_epochs, islands,
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

        # return score_tracker.all_time_best_fitness
        return best_fitness


class Island:
    def __init__(self, identifier, f, popul_size, n, mutation, crossover):
        self.identifier = identifier
        self.f = f
        self.popul_size = popul_size
        self.population = Island.initialize_population(self.popul_size, n)
        self.mutation = mutation
        self.crossover = crossover

    @staticmethod
    def initialize_population(population_size, num_cities):
        # returns `population_size` number of permutations of `num_cities` cities

        # if would also like to include 0, then do:
        # population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])

        # add 0 to the start of each individual (implicit starting point)
        # values between 1 and num_cities
        population = np.array([np.random.permutation(num_cities - 1) + 1 for _ in range(population_size)], dtype=int)

        # Our representation (adjacency representation):
        # eg: 0 1 2 3 4
        #     | | | | |
        #     v v v v v
        #    [2,3,4,0,1]
        # so order of which city to visit is 2,3,4,0,1

        # We didn't cycle-representation because not easy to ensure that starts and stops at 0

        return population

    @staticmethod
    def run_epochs(nb_epochs, islands, selection, elimination, fitness_sharing, local_search, score_tracker, ctr):
        # done_and_time_left = np.zeros((len(islands), ), dtype=bool)  # done for each island
        done_and_time_left_and_best_score = np.zeros((len(islands), 3), dtype=np.float64)  # done for each island

        done_and_time_left_and_best_score = [
            np.array(island._run_epoch(done_and_time_left_and_best_score[idx, 0], nb_epochs, idx, island, selection,
                                       elimination,
                                       fitness_sharing, local_search, score_tracker,
                                       ctr))
            for idx, island in enumerate(islands)]

        last_elt = done_and_time_left_and_best_score[-1]
        done = last_elt[0]
        time_left = last_elt[1]
        best_score = last_elt[2]
        return done, time_left, best_score

    def _run_epoch(self, done, nb_epochs, island_idx, island, selection, elimination, fitness_sharing, local_search,
                   score_tracker,
                   ctr):

        # _run_epoch is called for epoch amount of times, could be that time was already over in previous epoch
        # so don't run epoch if time is already over
        if done:
            return done

        # contains all fitnesses of a single island (for all epochs)
        best_fitnesses = np.zeros(nb_epochs, dtype=np.float64)
        mean_fitnesses = np.zeros(nb_epochs, dtype=np.float64)

        for epoch in range(nb_epochs):
            # overwrites best_fitness, mean_fitness, sigma_best, but that's ok to me
            best_fitnesses[epoch], mean_fitnesses[epoch], best_sigma, last_fitnesses_shared = island.step(
                selection, elimination, fitness_sharing, local_search, score_tracker, epoch + ctr)

            if epoch == nb_epochs - 1:  # only print results for last epoch of each island
                Utility.print_score((ctr * nb_epochs) + epoch, best_fitnesses[epoch], np.mean(mean_fitnesses), 1,
                                    avg_dist_func=lambda: FitnessSharing.avg_dist_func(island.population),
                                    fitnesses_shared=np.mean(last_fitnesses_shared),
                                    island=island)

            write_to_file = True if island_idx == 0 else False  # only write to file for first island
            done, time_left = score_tracker.utility.is_done_and_report(
                (ctr * nb_epochs) + epoch, mean_fitnesses[epoch], best_fitnesses[epoch], best_sigma,
                write_to_file=write_to_file)
            if done:
                break

        return done, time_left, best_fitnesses[
            epoch]  # I didn't dare -1, since if was asked to stop at eg 13, but 25 epochs...then would return 0

    def step(self, selection, elimination, fitness_sharing, local_search, score_tracker, ctr):
        fitnesses = self.f(self.population)  # before fitness sharing

        # Fitness sharing (MUST COME AFTER score_tracker.update_scores)
        fitnesses_shared = fitness_sharing(fitnesses, self.population)

        # Update scores
        best_fitness, mean_fitness, best_sigma = score_tracker.update_scores(
            fitnesses, self.population, ctr,
            fitnesses_shared=fitnesses_shared,
            pdf=None, print_w=False,  # pdf, w is only applicable to PlackettLuce, not Evol
            # only applicable to Evol, not PlackettLuce
            avg_dist_func=lambda: FitnessSharing.avg_dist_func(self.population),
            island_identifier=self.identifier,
            print_score=False  # printing is already done in _run_epoch, but print_score=True is used in PlackettLuce
        )

        # Selection
        selected = selection(self.population, fitnesses_shared)

        # Variation
        offspring = self.crossover(selected)
        # offspring = selected.copy() # no crossover
        self.mutation(offspring)

        offspring = local_search(offspring)

        joined_popul = np.vstack((offspring, self.population))  # old population should have been optimized before

        # Evaluation / elimination
        fitnesses = self.f(joined_popul)
        self.population = elimination(joined_popul, fitnesses)  # elimination is not based on fitness sharing

        # shuffle popul in place, required because other functions such
        # Diversity.fitness_sharing uses the first 10% of the population assuming it is random
        np.random.shuffle(self.population)

        # sanity check
        n = len(self.population[0])
        for i in range(len(self.population)):
            assert len(self.population[i]) == len(set(self.population[i])) == n

        # check if population still has the same size
        assert len(self.population) == self.popul_size

        return best_fitness, mean_fitness, best_sigma, fitnesses_shared

    @staticmethod
    def migrate(islands, popul_size, percentage=0.1):
        assert len(islands) > 0

        if len(islands) == 1:
            return

        print("Migrating...")
        # 10% of the population migrates to the next island
        migrants = islands[-1].population[:int(popul_size * percentage)]
        for idx, island in enumerate(islands):
            migrants = island._migrate(migrants)

    def _migrate(self, other_island_migrants):
        nb_migrants = len(other_island_migrants)
        our_migrants = self.population[:nb_migrants].copy()  # take first nb_migrants, already shuffled
        self.population[:nb_migrants] = other_island_migrants

        np.random.shuffle(self.population)

        return our_migrants

    @staticmethod
    def merge_islands(islands, crossover, mutation):
        assert len(islands) > 0

        if len(islands) == 1:
            return islands[0]

        # Must select the best individuals from each island
        entire_population = np.vstack([island.population for island in islands])
        entire_fitnesses = np.hstack([island.f(island.population) for island in islands])
        popul_size = islands[0].popul_size
        population = Selection.elimination(entire_population, entire_fitnesses, 3, popul_size)
        np.random.shuffle(population)

        # create an island with the merged population
        island = islands[0]
        island.population = population
        island.mutation = mutation
        island.crossover = crossover

        return island


class FitnessSharing:
    @staticmethod
    def distance(individual1, individual2):
        # individual1 and individual2 are in adjacency representation
        # so we need to convert to cycle representation, then can easily calculate distance by counting edges
        n = len(individual1) + 1
        indiv1_cyclic = Variation.edge_table(individual1, n)  # eg: 1 2 3 4 --> 1 2 3 4 0
        # add 0 to the start of each individual (implicit starting point)
        individual2 = np.insert(individual2, 0, 0)  # eg: 1 2 3 4 --> 0 1 2 3 4

        # indiv2: 0 1 2 3 4
        #         | | | | |
        #         v v v v v
        #         0 1 2 3 4

        # indiv1_cyclic: 1 2 3 4 0
        #                | | | | |
        #                v v v v v
        #                2 3 4 0 1

        nb_equal_edges = 0
        indiv1_points_to = indiv1_cyclic[0]
        for i in range(n):
            indiv2_points_to = individual2[(i + 1) % n]
            if indiv1_points_to == indiv2_points_to:
                nb_equal_edges += 1

            indiv1_points_to = indiv1_cyclic[indiv1_points_to]

        return n - nb_equal_edges

    @staticmethod
    def get_single_fitness_shared(org_fitness, population, subpopulation, sub_popul_size, sub_popul_percent, i,
                                  n, alpha):
        # A single individual must be compared to all individuals in the subpopulation

        # if i == j then distance is 0
        distances = [FitnessSharing.distance(population[i], subpopulation[j]) if not (i == j) else 0
                     for j in range(sub_popul_size)]

        if np.mean(distances) == 0:
            # print("mean distance is 0, SO IS COPY OF ALL.")
            a = 5

        sharing_vals = np.array([FitnessSharing.sharing_function(d=distances[j], max_distance=n, alpha=alpha)
                                 for j in range(sub_popul_size)])
        sharing_vals = 1 - sharing_vals  # element-wise

        sum_sharing_vals = np.sum(sharing_vals)  # dependent on the subpopulation sizedfsafds
        # So to rescale, we divide by the subpopulation percent
        sum_sharing_vals = sum_sharing_vals / sub_popul_percent

        # add 1 to the sharing val (sum) for the remaining 90% of the population (as explained in `fitness_sharing` func due to subpopul not including all individuals)
        sum_sharing_vals += 1 if i >= sub_popul_size else 0

        # 283626.35370461695
        fitness_shared = org_fitness * sum_sharing_vals

        # = 1 would be if all individuals are entirely different (highly unlikely)
        assert sum_sharing_vals >= 1
        assert fitness_shared >= org_fitness
        return fitness_shared, sum_sharing_vals  # the sum is the penalty for being close to another individual

    @staticmethod
    def fitness_sharing(fitnesses_org, population, sub_popul_percent, alpha):
        """
        The fitness sharing function is used to punish individuals that are close to other individuals.
        However, the fitness sharing function is computationally expensive, so we only compute it for the first 10% of the population.
        In addition for this 10%, we only compare it to the first 10% of the population, again for computational reasons. This gives an estimate of the real fitness sharing value.

        The remaining 90% of the population receives a sharing val of fitness * avg sharing val multiplier.
        """

        if sub_popul_percent == 0:  # if 0, then no fitness sharing
            return fitnesses_org

        popul_size = len(population)
        n = len(population[0])

        # randomly take 10% of the population to consider for sharing
        # sub_popul_percent = 0.1  # problem w/ subpopul is that for specific individuals, distance is 0 and for its very large
        # so can be that indiv is not included in neighbourhood so sharing val is 0. needs to be included in neighbourhood!
        sub_popul_size = int(popul_size * sub_popul_percent)

        subpopulation = population[:sub_popul_size]  # take first 10% of population
        # for remaining 90% of population, add 1 to the sharing val

        # since computationally expensive, only compute for the first 10% of the population, the rest receives a sharing val of fitness * avg sharing val multiplier
        fitnesses_shared_and_sum = np.zeros((popul_size, 2), dtype=np.float64)

        # pick the last 10% of the population (the first 10% is used to compare against: subpopulation, see get_single_fitness_shared)
        indices_steekproef = np.arange(popul_size - sub_popul_size,
                                       popul_size)  # of these the shared_fitness is actually computed
        fitnesses_shared_and_sum[indices_steekproef, :] = np.array(
            [FitnessSharing.get_single_fitness_shared(
                fitnesses_org[i],
                population,
                subpopulation,
                sub_popul_size,
                sub_popul_percent, i, n, alpha)
                for i in indices_steekproef])

        fitnesses_shared = fitnesses_shared_and_sum[:, 0]
        penalty_sums = fitnesses_shared_and_sum[:, 1]

        # of these the shared_fitness is not computed, and their shared_fitness is defined as `fitness * avg penalty sum`
        indices_avg = np.arange(popul_size - sub_popul_size)

        # the rest of the population receives a sharing val of fitness * avg sharing val multiplier
        avg_sum = np.mean(penalty_sums[indices_steekproef])
        fitnesses_shared[indices_avg] = fitnesses_org[indices_avg] * avg_sum

        return fitnesses_shared

    @staticmethod
    def sharing_function(d, max_distance, alpha):
        # sigma_share is based on the maximum distance between any two individuals in the population, which is n
        # so only punish a candidate solution if it has a neighbour that is 1% of the max distance away
        # with similarity = # edges in common
        # so if path is 750 cities, punish node if it has a neighbour w/ 7.5 edges in common
        # sigma_share = max_distance * 0.1
        # sigma_share = max_distance * 0.2  # half of max distance

        sigma_share = max_distance  # punish all individuals, how severe depends on alpha
        # alpha++ increases the penalty for being close to another individual
        # alpha-- decreases the penalty for being close to another individual

        assert sigma_share >= d

        if d <= sigma_share:
            val = (d / sigma_share) ** alpha
            return val
        else:
            # throw not supported exception
            raise Exception("d > sigma_share")
            return 0

    @staticmethod
    def avg_dist_func(population):
        """
        Approximation of the average distance between individuals in the population
        """
        average_distance = np.mean(
            [FitnessSharing.distance(population[i], population[i + 1]) for i in range(len(population) - 1)])
        return average_distance


class LocalSearch:

    @staticmethod
    def insert_random_node(population, d, nb_nodes_to_insert_percent=0.1):
        nb_nodes_to_insert = int(np.size(population, 1) * nb_nodes_to_insert_percent)
        # print(f"inserting {nb_nodes_to_insert} nodes")

        if nb_nodes_to_insert == 0:
            return population

        # verify perm_len: otherwise we would be inserting the same node
        max_perm_len = np.size(population, 1)
        assert nb_nodes_to_insert <= max_perm_len

        population = population.copy()
        nb_cities = np.size(population, 1)

        for indiv_idx, indivd in enumerate(population):
            # pick `nb_nodes_to_insert` random nodes without replacement
            rnd_a_indices = np.random.choice(nb_cities, nb_nodes_to_insert, replace=False)
            rnd_b_indices = np.random.choice(nb_cities, nb_nodes_to_insert, replace=False)

            for a_idx, b_idx in zip(rnd_a_indices, rnd_b_indices):
                if a_idx == b_idx:
                    continue

                # check if better performance
                a = indivd[a_idx]
                b = indivd[b_idx]
                a_next = indivd[(a_idx + 1) % nb_cities]
                b_next = indivd[(b_idx + 1) % nb_cities]
                b_prev = indivd[(b_idx - 1) % nb_cities]

                # ... a_prev -> a -> a_next -> ...
                # ... b_prev -> b -> b_next -> ...
                current_cost = d[a, a_next] + d[b_prev, b] + d[b, b_next]  # relevant cost of the original path

                # would become
                # ... a_prev -> a -> b -> b_next -> ...
                # ... b_prev -> b_next -> ...
                would_be_cost = d[a, b] + d[b, a_next] + d[b_prev, b_next]

                if current_cost > would_be_cost:
                    # print(f"inserting node {b} after node {a} in individual {indiv_idx}")
                    # insert node b after node a and shift all other nodes
                    indivd = np.insert(indivd, a_idx + 1, indivd[b_idx])
                    if a_idx < b_idx:
                        indivd = np.delete(indivd, b_idx + 1)
                    else:
                        indivd = np.delete(indivd, b_idx)

                    population[indiv_idx] = indivd

        return population

    @staticmethod
    def two_opt_slow(population, d):
        # https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
        population = population.copy()
        population = np.hstack((np.zeros((len(population), 1), dtype=int), population))
        nb_cities = np.size(population, 1)

        # since matrix is not symmetric, we need to check both directions
        for indiv_idx, indivd in enumerate(population):
            # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

            for i in range(nb_cities - 1):
                for j in range(0, nb_cities - 1):  # maybe can start at i+1 if we know the matrix is symmetric
                    if i == j:
                        continue

                    a = indivd[i]
                    b = indivd[j]
                    a_next = indivd[i + 1]
                    b_next = indivd[j + 1]

                    # take path from a to a_next and b to b_next and sum them
                    current_cost = d[a, a_next] + d[b, b_next]
                    would_be_cost = d[a, b] + d[a_next, b_next]
                    if current_cost > would_be_cost:
                        # reverse path from a_next to b
                        indivd[i + 1:j + 1] = indivd[j:i:-1]
                        # warning: this only works for matrices that are symmetric, as the reverse path is not the same

        # Note that the 0 node is not swapped, so we can remove it
        population = population[:, 1:]
        return population

    @staticmethod
    def two_opt(population, d, jump_size):
        # https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
        population = population.copy()
        population = np.hstack((np.zeros((len(population), 1), dtype=int), population))
        nb_cities = np.size(population, 1)

        for indiv_idx, indivd in enumerate(population):
            # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

            starting_idx1 = np.random.randint(0, jump_size)
            for i in range(starting_idx1, nb_cities - 1, jump_size):
                # starting_idx2 = np.random.randint(0, jump_size)
                # for j in range(starting_idx2, nb_cities - 1, jump_size):  # maybe can start at i+1 if we know the matrix is symmetric
                for j in range(nb_cities - 1):  # maybe can start at i+1 if we know the matrix is symmetric
                    if i == j:
                        continue

                    a = indivd[i]
                    b = indivd[j]
                    a_next = indivd[i + 1]
                    b_next = indivd[j + 1]

                    # take path from a to a_next and b to b_next and sum them
                    current_cost = d[a, a_next] + d[b, b_next]
                    would_be_cost = d[a, b] + d[a_next, b_next]
                    if current_cost > would_be_cost:
                        # reverse path from a_next to b
                        indivd[i + 1:j + 1] = indivd[j:i:-1]

        # Note that the 0 node is not swapped, so we can remove it
        population = population[:, 1:]
        return population

    @staticmethod
    def two_opt_swap(population, d):
        population = population.copy()
        population = np.hstack((np.zeros((len(population), 1), dtype=int), population))
        nb_cities = np.size(population, 1)

        # Since matrix is not symmetric, we need to check both directions
        for indiv_idx, indivd in enumerate(population):
            # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

            for i in range(nb_cities):
                for j in range(0, nb_cities):  # maybe can start at i+1 if we know the matrix is symmetric
                    if i == j:
                        continue

                    a = indivd[i]
                    b = indivd[j]
                    a_next = indivd[(i + 1) % nb_cities]
                    b_next = indivd[(j + 1) % nb_cities]

                    # take path from a to a_next and b to b_next and sum them
                    current_cost = d[a, a_next] + d[b, b_next]
                    would_be_cost = d[a, b] + d[a_next, b_next]
                    if current_cost > would_be_cost:
                        # a should point to b_next, and b should point to a_next
                        indivd[(i + 1) % nb_cities] = b
                        indivd[(j + 1) % nb_cities] = a

        population = population[:, 1:]

        return population


class Selection:
    @staticmethod
    def selection(population, k, nb_individuals_to_select, fitness_scores, allow_duplicates=True):
        if allow_duplicates:
            return Selection.selection_with_duplicates(population, k, nb_individuals_to_select, fitness_scores)
        else:
            return Selection.selection_without_duplicates(population, k, nb_individuals_to_select, fitness_scores)

    @staticmethod
    def selection_without_duplicates(population, k, nb_individuals_to_select, fitness_scores):
        popul_size = np.size(population, 0)
        assert nb_individuals_to_select <= popul_size - k + 1

        # deleted_individuals = np.bool_(np.zeros(len(population)))  # default: false
        deleted_individuals = np.zeros(len(population), dtype=bool)  # default: false
        nb_cities = np.size(population, 1)
        selected = np.zeros((nb_individuals_to_select, nb_cities), dtype=int)
        for ii in range(nb_individuals_to_select):
            # indices of random individuals
            plausible_indices = np.where(deleted_individuals == False)
            plausible_indices = plausible_indices[0]  # tuple i assume
            ri = np.random.choice(plausible_indices, k, replace=False)

            min = np.argmin(fitness_scores[ri])
            best_indiv_idx = ri[min]
            selected[ii, :] = population[best_indiv_idx, :]
            deleted_individuals[best_indiv_idx] = True

        return selected

    @staticmethod
    def selection_with_duplicates(population, k, nb_individuals_to_select, fitness_scores):
        nb_cities = np.size(population, 1)
        popul_size = np.size(population, 0)
        selected = np.zeros((nb_individuals_to_select, nb_cities), dtype=int)
        for ii in range(nb_individuals_to_select):
            # indices of random individuals
            # random.choices to prevent comparing 2 identical individuals
            # ri = random.choices(range(popul_size), k=k)
            ri = np.random.choice(range(popul_size), k, replace=True)
            min = np.argmin(fitness_scores[ri])  # take the single best
            best_indiv_idx = ri[min]
            selected[ii, :] = population[best_indiv_idx, :]
        return selected  # this may contain duplicates

    @staticmethod
    def elimination(joinedPopulation, fitness_scores, k, popul_size):
        # Not age based because loses potentially good individuals
        # just do selection again
        # In this case, sample without replacement. (selection was with replacement, so allowed duplicates)
        return Selection.selection(joinedPopulation, k, popul_size, fitness_scores, allow_duplicates=False)


class Variation:

    @staticmethod
    def crossover(selected):
        offspring = Variation.edge_crossover(selected)
        return offspring

    @staticmethod
    def single_cross_over_step(idx, curr_elt, father_ciclic, mother_ciclic, deleted_cities):
        # 4. Remove all references to current element from the table
        deleted_cities[curr_elt] = True

        # city = 5, edge list = {1, 6} so 5 -> 1
        # city = 5, edge list = {1, 1}

        # 5. Examine list for current element
        # • If there is a common edge, pick that to be the next element
        next_city1 = father_ciclic[curr_elt]
        next_city2 = mother_ciclic[curr_elt]
        if next_city1 == next_city2 and not deleted_cities[next_city1]:
            curr_elt = next_city1

        else:
            # • Otherwise pick the entry in the list which itself has the shortest list
            # check if one deleted, then pick other
            if deleted_cities[next_city1] and not deleted_cities[next_city2]:
                curr_elt = next_city2
            elif deleted_cities[next_city2] and not deleted_cities[next_city1]:
                curr_elt = next_city1
            elif not (deleted_cities[next_city1]) and not (
                    deleted_cities[next_city2]):  # so neither are deleted
                length_city1 = Variation.get_list_length(father_ciclic, mother_ciclic, next_city1, deleted_cities)
                length_city2 = Variation.get_list_length(father_ciclic, mother_ciclic, next_city2, deleted_cities)

                if length_city1 < length_city2:
                    curr_elt = next_city1
                elif length_city1 > length_city2:
                    curr_elt = next_city2
                else:
                    # np upperbound is exclusive
                    curr_elt = next_city1 if np.random.randint(0, 1 + 1) == 0 else next_city2
                    # curr_elt = next_city1 if random.randint(0, 1) == 0 else next_city2


            else:  # both are deleted so pick random
                available_cities = np.where(deleted_cities == False)
                available_cities = available_cities[0]  # cause its a tuple

                assert len(available_cities) > 0

                curr_elt = np.random.choice(available_cities)
                # curr_elt = random.choice(available_cities)

        # • Ties are split at random
        #   Just pick from the father, if common then its also part of mother.
        #   There is no randomness, but it is not needed I think since father and mother are random

        # city_points_to1 = father[curr_elt]
        # city_points_to2 = mother[curr_elt]
        # curr_elt = city_points_to1 if used_cities[city_points_to1] == 0 else city_points_to2

        # 6. In the case of reaching an empty list, the other end of the offspring is
        # examined for extension; otherwise a new element is chosen at random

        return curr_elt

    @staticmethod
    def edge_crossover(selected):
        """
        :return: offsprings in adjacency representation
        """
        nb_cities = np.size(selected, 1) + 1  # +1 because we need to add 0 at the start

        fathers = selected[::2]
        mothers = selected[1::2]

        # offsprings are in adjancency representation, the implicit 0 is not included
        offsprings = np.zeros((len(fathers), nb_cities - 1), dtype=int)

        nb_fathers = len(fathers)
        for i in range(nb_fathers):
            father = fathers[i]
            mother = mothers[i]

            # 1. Construct edge table
            father_ciclic = Variation.edge_table(father, nb_cities)  # implicit 0 is included
            mother_ciclic = Variation.edge_table(mother, nb_cities)

            deleted_cities = np.zeros(nb_cities, dtype=bool)  # default: false
            offspring = np.zeros(nb_cities, dtype=int)  # implicit 0 is included

            # 2. pick an initial elt at rnd and put it in the offspring
            # city = random.randint(0, nb_cities - 1)
            city = np.random.randint(0, nb_cities)  # np.random.randint upperboun is exclusive
            curr_elt = city

            # 3. Set the variable current element = entry
            offspring[0] = curr_elt

            for idx in range(nb_cities - 1):
                curr_elt = Variation.single_cross_over_step(idx, curr_elt, father_ciclic, mother_ciclic, deleted_cities)
                offspring[idx + 1] = curr_elt

            # offsprings now also contain the 0 somewhere in the middle
            # so we need to shift it to the start and remove the final element (which is 0)
            zero_idx = np.where(offspring == 0)[0][0]
            offspring = np.roll(offspring, shift=-zero_idx, axis=0)
            offspring = offspring[1:]

            offsprings[i, :] = offspring

        return offsprings

    @staticmethod
    def swap_mutation(offspring_popul, mutation_rate):
        """
        Swaps exactly 2 cities in the offspring
        :param offspring_popul: shape: (popul_size, nb_cities)
        :param mutation_rate: float
        :return:
        """

        # Calculate the number of individuals to mutate based on the mutation rate
        num_individuals_to_mutate = int(len(offspring_popul) * mutation_rate)

        # reshuffle the offspring
        np.random.shuffle(offspring_popul)

        # Select random indices to mutate
        offspring_popul = offspring_popul[:num_individuals_to_mutate]

        nb_cities = np.size(offspring_popul, 1)
        row_indices = np.arange(len(offspring_popul))
        idx1 = np.random.randint(0, nb_cities, len(offspring_popul))
        idx2 = np.random.randint(0, nb_cities, len(offspring_popul))

        # Swap elements using advanced indexing
        # idea: offspring_popul[0], offspring_popul[1] = offspring_popul[1], offspring_popul[0]
        offspring_popul[row_indices, idx1], offspring_popul[row_indices, idx2] = \
            offspring_popul[row_indices, idx2], offspring_popul[row_indices, idx1]

    @staticmethod
    def inversion_mutation(offspring_popul, mutation_rate):  # Entirely generated via Copilot
        """
        Inverts a random subsequence of the offspring
        :param offspring_popul: shape: (popul_size, nb_cities)
        :param mutation_rate: float
        :return:
        """

        # Calculate the number of individuals to mutate based on the mutation rate
        num_individuals_to_mutate = int(len(offspring_popul) * mutation_rate)

        # reshuffle the offspring
        np.random.shuffle(offspring_popul)

        # Select random indices to mutate
        offspring_popul = offspring_popul[:num_individuals_to_mutate]

        nb_cities = np.size(offspring_popul, 1)
        row_indices = np.arange(len(offspring_popul))
        idx1 = np.random.randint(0, nb_cities, len(offspring_popul))
        idx2 = np.random.randint(0, nb_cities, len(offspring_popul))

        # Swap elements using advanced indexing
        # idea: offspring_popul[0], offspring_popul[1] = offspring_popul[1], offspring_popul[0]
        offspring_popul[row_indices, idx1], offspring_popul[row_indices, idx2] = \
            offspring_popul[row_indices, idx2], offspring_popul[row_indices, idx1]

    @staticmethod
    def scramble_mutation(offspring_popul, mutation_rate):  # Entirely generated via Copilot
        """
        Scrambles a random subsequence of the offspring
        :param offspring_popul: shape: (popul_size, nb_cities)
        :param mutation_rate: float
        :return:
        """

        # Calculate the number of individuals to mutate based on the mutation rate
        num_individuals_to_mutate = int(len(offspring_popul) * mutation_rate)

        # reshuffle the offspring
        np.random.shuffle(offspring_popul)

        # Select random indices to mutate
        offspring_popul = offspring_popul[:num_individuals_to_mutate]

        nb_cities = np.size(offspring_popul, 1)
        row_indices = np.arange(len(offspring_popul))
        idx1 = np.random.randint(0, nb_cities, len(offspring_popul))
        idx2 = np.random.randint(0, nb_cities, len(offspring_popul))

        # Swap elements using advanced indexing
        # idea: offspring_popul[0], offspring_popul[1] = offspring_popul[1], offspring_popul[0]
        offspring_popul[row_indices, idx1], offspring_popul[row_indices, idx2] = \
            offspring_popul[row_indices, idx2], offspring_popul[row_indices, idx1]

    @staticmethod
    def edge_table(parent, nb_cities):
        assert np.size(parent) == nb_cities - 1  # -1 because we need to add 0 at the start
        # parent:
        # 0 1 2 3
        # | | | |
        # v v v v
        # 0 2 1 3

        # parent_cyclic:
        # 0 1 2 3
        # | | | |
        # v v v v
        # 2 3 1 0

        # Create edge table (is same as cycle representation)
        parent_ciclic = np.zeros(nb_cities, dtype=int)
        parent_ciclic[0] = int(parent[0])
        current_city_parent_ciclic = int(parent[0])

        # iterate over parent and add to edge table
        for i in range(nb_cities - 1):
            try:
                parent_ciclic[current_city_parent_ciclic] = int(parent[i])  # TODO: unsure if % is needed
                current_city_parent_ciclic = int(parent[i])
            except:
                print("parent:", parent)
                print("parent_ciclic:", parent_ciclic)
                print("current_city_parent_ciclic:", current_city_parent_ciclic)
                print("i:", i)
                raise

        return parent_ciclic

    @staticmethod
    def get_list_length(father_ciclic, mother_ciclic, curr_elt, deleted_cities):
        next_city_f = father_ciclic[curr_elt]
        next_city_m = mother_ciclic[curr_elt]
        count = 0
        if not (deleted_cities[next_city_f]):  # deleted -> 0
            count += 1
        if not (deleted_cities[next_city_m]) and next_city_m != next_city_f:  # deleted -> 0
            count += 1
        return count

    @staticmethod
    def order_crossover(selected):
        """ Calls crossover function twice for each pair of parents. Thus returns 2 offsprings per pair of parents
        :param selected: shape: (popul_size, nb_cities)
        :return: shape: (popul_size, nb_cities)
        """
        # add 0 node at the start
        selected = selected.copy()
        selected = np.hstack((np.zeros((len(selected), 1), dtype=int), selected))
        nb_cities = np.size(selected, 1)

        fathers = selected[::2]
        mothers = selected[1::2]

        offsprings1 = Variation._order_crossover(fathers, mothers, nb_cities)
        offsprings2 = Variation._order_crossover(mothers, fathers, nb_cities)
        offsprings = np.vstack((offsprings1, offsprings2))
        return offsprings

    @staticmethod
    def _order_crossover(fathers, mothers, nb_cities):
        # 1. Select two random crossover points
        idx1 = np.random.randint(0, nb_cities, len(fathers), dtype=np.int32)
        idx2 = np.random.randint(0, nb_cities, len(mothers), dtype=np.int32)
        idx1, idx2 = np.sort([idx1, idx2], axis=0)

        # 2. Copy the subsequence between the two points from the first parent to the first offspring
        # offsprings = np.zeros((len(fathers), nb_cities), dtype=int)
        offsprings = np.zeros_like(fathers, dtype=int)
        nb_fathers = len(fathers)
        for i in range(nb_fathers):
            offspring = offsprings[i]
            father = fathers[i]

            # copy the subsequence between the two points from the first parent to the first offspring
            offspring[idx1[i]:idx2[i]] = father[idx1[i]:idx2[i]]

        # 3. Copy the remaining elements from the second parent to the first offspring, starting after the second crossover point, wrapping around the list
        # must check if the element is already in the offspring, if so then skip
        nb_mothers = len(mothers)

        offsprings_wo_zeros = np.zeros((nb_mothers, nb_cities - 1), dtype=int)
        for i in range(nb_mothers):
            offspring = offsprings[i]
            mother = mothers[i]

            idx_mother = idx2[i]  # start copying from the second crossover point
            idx_offspring = (idx2[i] + 1) % nb_cities
            # copy the remaining elements from the second parent to the first offspring
            for j in range(nb_cities):
                if mother[idx_mother] not in offspring:
                    offspring[idx_offspring] = mother[idx_mother]
                    idx_offspring = (idx_offspring + 1) % nb_cities
                idx_mother = (idx_mother + 1) % nb_cities

            # Must remove the 0 from the offspring and shift the elements to the start
            zero_idx = np.where(offspring == 0)[0][0]  # first find the index of the 0
            offspring = np.roll(offspring, shift=-zero_idx, axis=0)
            offspring = offspring[1:]
            offsprings_wo_zeros[i] = offspring

        return offsprings_wo_zeros


class ScoreTracker:
    def __init__(self, n, maximize, keep_running_until_timeup, numIters, reporter_name, benchmark, second_filename,
                 max_duration=None):
        self.maximize = maximize
        self.all_time_best_fitness = -np.inf if maximize else np.inf
        self.all_time_sigma_best = np.zeros(n, dtype=np.int64)
        reporter = Reporter.Reporter(reporter_name, second_filename, max_duration=max_duration)
        self.utility = Utility(reporter, keep_running_until_timeup, numIters)
        self.benchmark = benchmark

    def update_scores(self, fitnesses, sigmas, ctr, fitnesses_shared=None, pdf=None, print_w=False, avg_dist_func=None,
                      island_identifier=None, print_score=True):

        fitnesses = self.benchmark.unnormalize_fitnesses(fitnesses)
        if fitnesses_shared is not None:
            fitnesses_shared = self.benchmark.unnormalize_fitnesses(fitnesses_shared)

        # code is clearer
        if self.maximize:
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            sigma_best = sigmas[best_idx]
            if best_fitness > self.all_time_best_fitness:
                self.all_time_best_fitness = best_fitness
                self.all_time_sigma_best = sigma_best
        else:
            best_idx = np.argmin(fitnesses)
            best_fitness = fitnesses[best_idx]
            sigma_best = sigmas[best_idx]
            if best_fitness < self.all_time_best_fitness:
                self.all_time_best_fitness = best_fitness
                self.all_time_sigma_best = sigma_best

        if print_w and pdf is not None:
            assert island_identifier is None

            w = np.exp(pdf.w_log)
            frequency = 100
            if len(w.shape) == 2:  # if w_log is square matrix:
                Utility.print_mtx(w, ctr, frequency, sub_mtx=10)
            elif len(w.shape) == 1:  # if w_log is 1d array:
                Utility.print_array(w, ctr, frequency)
            else:
                raise Exception("w_log has unsupported shape")

        avg_fitness = np.mean(fitnesses)

        if print_score:
            Utility.print_score(ctr, best_fitness, avg_fitness, 10, avg_dist_func, fitnesses_shared,
                                island_identifier)
        return best_fitness, avg_fitness, sigma_best


class Reporter:
    def __init__(self, filename, second_filename=None, max_duration=None):
        print("Reporter: " + filename)
        # self.allowedTime = 300 #5 minutes
        self.allowedTime = max_duration if max_duration is not None else 300
        self.numIterations = 0
        self.filename = filename + ".csv"
        self.second_filename = second_filename + ".csv" if second_filename is not None else None
        self.delimiter = ','
        self.startTime = time.time()
        self.writingTime = 0
        outFile = open(self.filename, "w")
        outFile.write("# Student number: " + filename + "\n")
        outFile.write("# Iteration, Elapsed time, Mean value, Best value, Cycle\n")
        outFile.close()

        if self.second_filename is not None:
            outFile = open(self.second_filename, "w")
            outFile.write("# Student number: " + second_filename + "\n")
            outFile.write("# Iteration, Elapsed time, Mean value, Best value, Cycle\n")
            outFile.close()

    # Append the reported mean objective value, best objective value, and the best tour
    # to the reporting file.
    #
    # Returns the time that is left in seconds as a floating-point number.
    def report(self, meanObjective, bestObjective, bestSolution, write_to_file=True):
        if (time.time() - self.startTime < self.allowedTime + self.writingTime) and write_to_file:
            start = time.time()
            outFile = open(self.filename, "a")
            outFile.write(str(self.numIterations) + self.delimiter)
            outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
            outFile.write(str(meanObjective) + self.delimiter)
            outFile.write(str(bestObjective) + self.delimiter)
            for i in range(bestSolution.size):
                outFile.write(str(bestSolution[i]) + self.delimiter)
            outFile.write('\n')
            outFile.close()

            # It is obviously a bit silly to write the same thing twice, but I didn't dare to change writing to
            # r0698535.csv, as it may be used by the professor to grade the assignment.
            if self.second_filename is not None:
                outFile = open(self.second_filename, "a")
                outFile.write(str(self.numIterations) + self.delimiter)
                outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
                outFile.write(str(meanObjective) + self.delimiter)
                outFile.write(str(bestObjective) + self.delimiter)
                for i in range(bestSolution.size):
                    outFile.write(str(bestSolution[i]) + self.delimiter)
                outFile.write('\n')
                outFile.close()

            self.numIterations += 1
            self.writingTime += time.time() - start
        return (self.allowedTime + self.writingTime) - (time.time() - self.startTime)


class HyperparamsEvolAlgorithm:
    def __init__(self,
                 popul_size=100,
                 offspring_size_multiplier=1,
                 k=3,
                 mutation_rate=0.2,
                 # Islands
                 migrate_after_epochs=25, migration_percentage=0.1, merge_after_percent_time_left=0.5,
                 fitness_sharing_subset_percentage=0.1,  # higher is more accurate, but slower
                 alpha=1,  # used in fitness sharing
                 local_search=(None, None),
                 keep_running_until_timeup=True):
        self.popul_size = popul_size
        self.offspring_size_multiplier = offspring_size_multiplier
        self.k = k
        self.mutation_rate = mutation_rate

        self.nb_islands = 3  # Always fixed. one island per mutation function
        self.migrate_after_epochs = migrate_after_epochs
        self.migration_percentage = migration_percentage
        self.merge_after_percent_time_left = merge_after_percent_time_left  # eg 0.75 will merge when 75% of time is left

        self.fitness_sharing_subset_percentage = fitness_sharing_subset_percentage
        self.alpha = alpha

        self.local_search = local_search  # (None, None), ("2-opt", 1), ("insert_random_node", 0.1) ...
        # 2nd param is param for local search:
        # eg nb_nodes_to_insert_percent=0.1 for local_search="insert_random_node", jump_size=1 for local_search="2-opt"

        self.keep_running_until_timeup = keep_running_until_timeup


class HyperparamsPlackettLuceAlgorithm:
    def __init__(self,
                 pdf,
                 lr=0.9,
                 nb_samples_lambda=100,
                 U=PlackettLuce.U_identity,
                 keep_running_until_timeup=True):
        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda
        self.U = U
        self.keep_running_until_timeup = keep_running_until_timeup
        self.pdf: PdfRepresentation = pdf
        # pdf: PdfRepresentation = ConditionalPdf(benchmark.permutation_size())
        # algorithm = PlackettLuceAlgorithm(lr, nb_samples_lambda, U, benchmark, pdf)


class AlgorithmWrapper:
    def __init__(self, algorithm, numIters, max_duration=None):
        self.algorithm = algorithm
        self.numIters = numIters
        self.max_duration = max_duration

    @staticmethod
    def run_experiment_plackett_luce(hyperparams, benchmark_filename, pdf, reporter_name):
        print("*******************************************************************")
        print("Running experiment with parameters:")
        print(hyperparams.__dict__)

        # csv_filename is based on hyperparams and benchmark_filename
        GraphPlotter.mkdir(f"./pl/{benchmark_filename[:-4]}")
        csv_filename = (f"./pl/{benchmark_filename[:-4]}/lr={hyperparams.lr},"
                        f"nb_samples_lambda={hyperparams.nb_samples_lambda},"
                        f"U={hyperparams.U.__name__}")

        numIters = np.inf
        benchmark = Benchmark(benchmark_filename, normalize=True, maximise=False)

        algorithm = PlackettLuceAlgorithm(hyperparams.lr, hyperparams.nb_samples_lambda, hyperparams.U, benchmark, pdf,
                                          hyperparams.keep_running_until_timeup, csv_filename)
        # a = r0698535.r0698535(algorithm, numIters, max_duration=60)  # 1 minute
        a = AlgorithmWrapper(algorithm, numIters, max_duration=60)  # 1 minute

        try:
            best_fitness = a.optimize(reporter_name)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            best_fitness = 0
        finally:
            GraphPlotter.read_file_and_make_graph(f"{csv_filename}.csv")

        return best_fitness

    @staticmethod
    def run_experiment(hyperparams, benchmark_filename, reporter_name):
        print("*******************************************************************")
        print("Running experiment with parameters:")
        print(hyperparams.__dict__)

        # csv_filename is based on hyperparams and benchmark_filename
        GraphPlotter.mkdir(f"./{benchmark_filename[:-4]}")
        csv_filename = (f"./{benchmark_filename[:-4]}/popul_size={hyperparams.popul_size},"
                        f"offsp_sz_multipl={hyperparams.offspring_size_multiplier},k={hyperparams.k},"
                        f"mut_r={hyperparams.mutation_rate},nb_isl={hyperparams.nb_islands},"
                        f"migr_aftr_ep={hyperparams.migrate_after_epochs},migr_perc={hyperparams.migration_percentage},"
                        f"mrge_aftr_perc_time_left={hyperparams.merge_after_percent_time_left},"
                        f"fit_shr_sbst_perc={hyperparams.fitness_sharing_subset_percentage},alph={hyperparams.alpha},"
                        f"local_search={hyperparams.local_search}")

        numIters = np.inf
        benchmark = Benchmark(benchmark_filename, normalize=False, maximise=False)

        algorithm = EvolAlgorithm(benchmark, hyperparams, csv_filename)
        # a = r0698535.r0698535(algorithm, numIters)
        a = AlgorithmWrapper(algorithm, numIters)

        try:
            best_fitness = a.optimize(reporter_name)
            # best_fitness = 0
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            best_fitness = 0
        finally:
            # plot
            GraphPlotter.read_file_and_make_graph(f"{csv_filename}.csv")

        return best_fitness

    @staticmethod
    def find_optimal_param(param_name, param_values, hyperparams, benchmark_filename, reporter_name):
        # *** POPUL_SIZE ***
        best_fitness = np.inf
        best_param = None
        for param_value in param_values:
            try:
                exec(f"hyperparams.{param_name} = {param_value}")

                fitness = AlgorithmWrapper.run_experiment(hyperparams, benchmark_filename, reporter_name)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_param = param_value
            except Exception as e:
                AlgorithmWrapper.append_to_file("best_params.txt", f"Error with {param_name} = {param_value}")
                # print hyper_params
                AlgorithmWrapper.append_to_file("best_params.txt", str(hyperparams.__dict__))
                AlgorithmWrapper.append_to_file("best_params.txt", str(e))
                continue

        # set best param
        exec(f"hyperparams.{param_name} = {best_param}")

        return best_param, best_fitness

    @staticmethod
    def clear_file(filename):
        with open(filename, "w") as f:
            f.write("")

    @staticmethod
    def append_to_file(filename, text):
        with open(filename, "a") as f:
            f.write(text + "\n")

    @staticmethod
    def find_optimal_param_for_tsp(benchmark_filename, reporter_name, fixed_popul_size=False):
        # Set parameters
        hyperparams = HyperparamsEvolAlgorithm()  # start with default params, and change one at a time

        test_params = {
            "popul_size": [10, 100, 200, 500, 1000] if not (fixed_popul_size) else [fixed_popul_size],
            "offspring_size_multiplier": [1, 2, 3],
            "k": [3, 5, 25],
            "mutation_rate": [0.05, 0.2, 0.4],
            "migrate_after_epochs": [25, 50],
            "migration_percentage": [0.05, 0.1],
            "merge_after_percent_time_left": [0.5, 0.75, 0.9],
            "fitness_sharing_subset_percentage": [0.05, 0.2, 0.5],
            "alpha": [1, 2, 0.5],
            "local_search": [(None, None), ("2-opt", 1), ("2-opt", 5),
                             ("insert_random_node", 0.1), ("insert_random_node", 0.5), ("insert_random_node", 1)]
        }

        # filename
        AlgorithmWrapper.append_to_file(f"best_params.txt", f"\n\n\n*********{benchmark_filename}*********")

        for param_name, param_values in test_params.items():
            best_param, all_time_best_fitness = AlgorithmWrapper.find_optimal_param(param_name, param_values,
                                                                                    hyperparams,
                                                                                    benchmark_filename, reporter_name)
            print()
            print()
            print()
            print("*" * 100)
            print(f"Best {param_name} is {best_param} with fitness {all_time_best_fitness}")
            print("*" * 100)
            print()
            print()
            print()
            AlgorithmWrapper.append_to_file("best_params.txt",
                                            f"Best {param_name} is {best_param} with fitness {all_time_best_fitness}")

    @staticmethod
    def repeat_experiment(hyperparams, benchmark_filename, reporter_name, nb_repeats=5, max_duration=15,
                          bar_chart=False):  # duration in seconds
        print("*******************************************************************")
        print("Running experiment with parameters:")
        print(hyperparams.__dict__)
        best_fitness = 0

        for i in range(nb_repeats):
            # csv_filename is based on hyperparams and benchmark_filename
            nb_tours = benchmark_filename[6:-4]
            GraphPlotter.mkdir(f"./BARS/{nb_tours}_tours/")
            csv_filename = (f"./BARS/{nb_tours}_tours/iter={i}")

            numIters = np.inf
            benchmark = Benchmark(benchmark_filename, normalize=False, maximise=False)

            algorithm = EvolAlgorithm(benchmark, hyperparams, csv_filename)
            # a = r0698535.r0698535(algorithm, numIters, max_duration=max_duration)
            a = AlgorithmWrapper(algorithm, numIters, max_duration=max_duration)

            try:
                best_fitness = a.optimize(reporter_name)
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                best_fitness = 0
            finally:
                GraphPlotter.read_file_and_make_graph(f"{csv_filename}.csv")
                pass

        if bar_chart:
            # only for 50 tours
            assert benchmark_filename == f"./tour{nb_tours}.csv"
            # after the nb_repeats, make a bar graph
            GraphPlotter.make_bar_graph(f"./BARS/{nb_tours}_tours", nb_repeats)

        return best_fitness

    def optimize(self, reporter_name):
        return self.algorithm.optimize(self.numIters, reporter_name, self.max_duration)


class GraphPlotter:
    # @staticmethod
    # def graph_path():
    #     graphs_path = "./graphs/"
    #     return graphs_path

    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def create_line_graph(x, y1, y2, xticks, x_label, y_label, title, filename=None):
        # xticks to integers
        xticks = [int(xtick) for xtick in xticks]

        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        # set max y value to 0.5
        # plt.ylim(top=100_000, bottom=0)

        # legend: best and mean
        plt.legend(["Mean objective value", "Best objective value"])

        # Set the x-axis ticks with a subset of xticks
        num_xticks_to_display = 10  # You can adjust this number
        step = len(xticks) // num_xticks_to_display
        # plt.xticks(x[::step], xticks[::step])

        if filename is not None:
            filename = filename  # GraphPlotter.graph_path() + filename
            plt.savefig(filename + ".pdf")
            plt.savefig(filename + ".png")
            # tikzplotlib.save(filename + ".tex")

        # plt.show()
        plt.close()  # close the figure, so it does not appear in the next graph

    @staticmethod
    def compare_best(x, ys, xticks, x_label, y_label, title, filename=None):
        # GraphPlotter.mkdir()

        # xticks to integers
        xticks = [int(xtick) for xtick in xticks]

        for y in ys:
            y = y[:len(x)]
            plt.plot(x, y)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        # set max y value to 0.5
        plt.ylim(top=100_000, bottom=0)

        # Set the x-axis ticks with a subset of xticks
        num_xticks_to_display = 10  # You can adjust this number
        step = len(xticks) // num_xticks_to_display
        plt.xticks(x[::step], xticks[::step])

        if filename is not None:
            filename = filename  # GraphPlotter.graph_path() + filename
            plt.savefig(filename + ".pdf")
            # tikzplotlib.save(filename + ".tex")
            plt.savefig(filename + ".png")

        plt.show()

    @staticmethod
    def read_report_file(filename):
        # GraphPlotter.mkdir()

        data = []

        numIterationss = []
        timeElapseds = []
        meanObjectives = []
        bestObjectives = []

        with open(filename, "r") as inFile:
            for line in inFile:
                parts = line.strip().split()  # Split the line into parts based on delimiter
                parts = parts[0]
                parts = parts.split(',')
                if len(parts) >= 5:  # Ensure that there are enough elements to extract
                    numIterations = int(parts[0])
                    timeElapsed = float(parts[1])
                    meanObjective = float(parts[2])
                    bestObjective = float(parts[3])

                    numIterationss.append(numIterations)
                    timeElapseds.append(timeElapsed)
                    meanObjectives.append(meanObjective)
                    bestObjectives.append(bestObjective)

        return numIterationss, timeElapseds, meanObjectives, bestObjectives

    @staticmethod
    def read_file_and_make_graph(filename="r0123456.csv", target_dir="./graphs/"):
        target_file = f"{target_dir}/full/{filename}"  # eg ./graphs/tour50/r0123456.csv
        target_file_skip_25 = f"{target_dir}/skip_first_25/{filename}"
        target_file_first_10_percent = f"{target_dir}/first_10_percent/{filename}"

        dir = os.path.dirname(target_file)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)

        results = GraphPlotter.read_report_file(filename)
        numIterationss, timeElapseds, meanObjectives, bestObjectives = results

        GraphPlotter.create_line_graph(
            numIterationss, meanObjectives, bestObjectives, timeElapseds,
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time",
            f"{target_file}_mean_objective_value")

        dir = os.path.dirname(target_file_skip_25)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)
        GraphPlotter.create_line_graph(
            numIterationss[25:], meanObjectives[25:], bestObjectives[25:], timeElapseds[25:],
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time",
            f"{target_file_skip_25}_mean_objective_value")

        dir = os.path.dirname(target_file_first_10_percent)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)
        GraphPlotter.create_line_graph(
            numIterationss[:int(len(numIterationss) * 0.1)],
            meanObjectives[:int(len(meanObjectives) * 0.1)],
            bestObjectives[:int(len(bestObjectives) * 0.1)],
            timeElapseds[:int(len(timeElapseds) * 0.1)],
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time (first 10% of iterations)",
            f"{target_file_first_10_percent}_mean_objective_value")

    @staticmethod
    # GraphPlotter.make_bar_graph(f"./BARS/50_tours/", nb_repeats)
    def make_bar_graph(dir, nb_repeats):
        """ Loads the files in dir (there are `nb_repeats` of them). Naming convention = iter=0.csv, iter=1.csv, ...
            Makes a histogram of the final mean fitnessess and the final best fitnesses of the `nb_repeats` runs.
        """
        mean_fitnesses = []
        best_fitnesses = []
        for i in range(nb_repeats):
            filename = f"{dir}/iter={i}.csv"
            # results = GraphPlotter.read_report_file(filename)
            # numIterationss, timeElapseds, meanObjectives, bestObjectives = results
            # mean_fitnesses.append(meanObjectives[-1])
            # best_fitnesses.append(bestObjectives[-1])

            # only read the last line of the file
            with open(filename, "r") as inFile:
                # get last line
                last_line = inFile.readlines()[-1]
                # split by comma
                parts = last_line.strip().split(',')
                # get mean and best fitness
                mean_fitness = float(parts[2])
                best_fitness = float(parts[3])
                # add to list
                mean_fitnesses.append(mean_fitness)
                best_fitnesses.append(best_fitness)

        # make the bar graph
        mean_fitnesses = np.array(mean_fitnesses)
        best_fitnesses = np.array(best_fitnesses)

        # print mean and standard deviation for mean and best fitnesses
        print(f"Mean fitnesses: {np.mean(mean_fitnesses)} +- {np.std(mean_fitnesses)}")
        print(f"Best fitnesses: {np.mean(best_fitnesses)} +- {np.std(best_fitnesses)}")

        fig, ax = plt.subplots()
        # set title
        ax.set_title(f"The final mean/best fitness for {nb_repeats} runs")

        ax.set_xlabel("Fitness")
        ax.set_ylabel("Frequency")
        bins = np.linspace(25_000, 38_000, 100)
        ax.hist(best_fitnesses, bins=bins, alpha=0.5)
        ax.hist(mean_fitnesses, bins=bins, alpha=0.5)
        # ax.hist(mean_fitnesses, bins=20)
        # ax.hist(best_fitnesses, bins=20)

        # plt.title(f"Final mean/best fitnesses of {nb_repeats} runs")
        # plt.xlabel("Fitness")
        # plt.ylabel("Frequency")
        # plt.hist(mean_fitnesses, bins=20)
        # plt.hist(best_fitnesses, bins=20, alpha=0.5)

        # plt.legend(["Mean fitnesses", "Best fitnesses"])

        # add mean and std to plot
        # plt.text(0.5, 0.5, f"Mean fitnesses: {np.mean(mean_fitnesses):.2f} +- {np.std(mean_fitnesses):.2f}\n \n"
        #                     f"Best fitnesses: {np.mean(best_fitnesses):.2f} +- {np.std(best_fitnesses):.2f}",
        #           horizontalalignment='center',
        #           verticalalignment='center',
        #           transform=plt.gca().transAxes)

        textstr = '\n'.join((
            r'$means=%.2f \pm %.2f$' % (np.mean(mean_fitnesses), np.std(mean_fitnesses),),
            r'$bests=%.2f \pm %.2f$' % (np.mean(best_fitnesses), np.std(best_fitnesses),),
        ))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        # legend
        # ax.legend(["Mean fitnesses", "Best fitnesses"])

        # legend bottom right
        ax.legend(["Mean fitnesses", "Best fitnesses"], loc='lower right')

        plt.savefig(f"{dir}/mean_fitnesses.png")
        plt.savefig(f"{dir}/mean_fitnesses.pdf")

        plt.show()
        plt.close()


class r0698535:
    def __init__(self):
        self.reporter_name = self.__class__.__name__
        self.run_plackett_luce = False

    def optimize(self, filename):
        if self.run_plackett_luce:
            assert filename.endswith("tour50.csv"), "PL-GS only supports tour50.csv"

            benchmark_filename = filename
            pdf: PdfRepresentation = VanillaPdf(n=50)
            # pdf: PdfRepresentation = ConditionalPdf(n=50)
            hyperparams = HyperparamsPlackettLuceAlgorithm(pdf)
            return AlgorithmWrapper.run_experiment_plackett_luce(hyperparams, benchmark_filename, pdf,
                                                                 self.reporter_name)

        else:  # run evol algorithm
            if filename.endswith("tour50.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                # *****./tour50.csv********* BEST PARAMS *****
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1  # 3
                hyperparams.k = 3
                hyperparams.mutation_rate = 0.1
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("2-opt", 1)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=20)  # only runs for 20 seconds, no need for longer

            elif filename.endswith("tour100.csv") or filename.endswith("tour200.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 3  # strange ...
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("2-opt", 1)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)  # will run for 5 minutes (default)
            elif filename.endswith("tour500.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 10
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)

            elif filename.endswith("tour750.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 25  # strange ...
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)  # will run for 5 minutes (default)

            else:  # for 1000 tours or any other file
                print("*" * 100)
                print("RUNNING OPTIMAL CONFIG FOR 1_000 TOURS!!!!")
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 25  # strange ... selection pressure is too high (popul is 50)
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 5
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)


if __name__ == "__main__":
    r = r0698535()
    print(r.optimize("./tour50.csv"))
    # print(r.optimize("./tour750.csv"))
    # print(r.optimize("./tour1000.csv"))
    # print(r.optimize("./tour500.csv"))
