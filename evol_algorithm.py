# import random

from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from placket_luce import PlackettLuce

import numpy as np

from variation import Variation


class EvolAlgorithm(AbstractAlgorithm):
    def __init__(self, benchmark, popul_size=1000, offspring_size_multiplier=2, k=3, mutation_rate=0.05):
        self.benchmark = benchmark

        self.popul_size = popul_size
        self.offspring_size = offspring_size_multiplier * popul_size
        self.k = k  # Tournament selection
        self.mutation_rate = mutation_rate
        self.keep_running_until_timeup = True

        super().__init__()

    def optimize(self, numIters, keep_running_until_timeup, reporter_name, *args):
        n = self.benchmark.permutation_size()

        # f = self.benchmark.compute_fitness
        # since zero is implicit, we need to the edge from 0 to first node
        f = lambda population: self.benchmark.compute_fitness(population) + self.benchmark.matrix[0, population[:, 0]]

        maximize = self.benchmark.maximise

        keep_running_until_timeup = keep_running_until_timeup
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.benchmark)

        # Initialize population
        population = self.initialize_population(self.popul_size, n)
        ctr = 0
        while True:

            fitnesses_not_scaled = f(population)  # before fitness sharing

            # Fitness sharing (MUST COME AFTER score_tracker.update_scores)
            fitnesses = self.fitness_sharing(fitnesses_not_scaled, population)

            # Update scores
            best_fitness, mean_fitness, sigma_best = score_tracker.update_scores(
                fitnesses_not_scaled, population, ctr,
                fitnesses_shared=fitnesses,
                pdf=None, print_w=False,  # pdf, w is only applicable to PlackettLuce, not Evol
                avg_dist_func=lambda: self.avg_dist_func(population)  # only applicable to Evol, not PlackettLuce
            )

            # Selection
            selected = self.selection(population, self.k, self.offspring_size, fitnesses)

            # Variation
            offspring = Variation.crossover(selected)
            Variation.mutation(offspring, self.mutation_rate)  # overwrites the offspring
            joined_popul = np.vstack((offspring, population))

            # Evaluation / elimination
            fitnesses = f(joined_popul)
            population = self.elimination(joined_popul, fitnesses)

            # shuffle population
            np.random.shuffle(population)

            # sanity check
            # for i in range(len(population)):
            #     assert len(population[i]) == len(set(population[i])) == n - 1

            ctr += 1
            if score_tracker.utility.is_done_and_report(ctr, mean_fitness, best_fitness, sigma_best):
                break

        return score_tracker.all_time_best_fitness

    def initialize_population(self, population_size, num_cities):
        # returns `population_size` number of permutations of `num_cities` cities

        # if would also like to include 0, then do:
        # population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])

        # add 0 to the start of each individual (implicit starting point)
        # values between 1 and num_cities
        population = np.array([np.random.permutation(num_cities - 1) + 1 for _ in range(population_size)])

        # Our representation (adjacency representation):
        # eg: 0 1 2 3 4
        #     | | | | |
        #     v v v v v
        #    [2,3,4,0,1]
        # so order of which city to visit is 2,3,4,0,1

        # We didn't cycle-representation because not easy to ensure that starts and stops at 0

        return population

    def selection(self, population, k, nb_individuals_to_select, fitness_scores, allow_duplicates=True):
        if allow_duplicates:
            return self.selection_with_duplicates(population, k, nb_individuals_to_select, fitness_scores)
        else:
            return self.selection_without_duplicates(population, k, nb_individuals_to_select, fitness_scores)

    def selection_without_duplicates(self, population, k, nb_individuals_to_select, fitness_scores):
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

    def selection_with_duplicates(self, population, k, nb_individuals_to_select, fitness_scores):
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

    def elimination(self, joinedPopulation, fitness_scores):
        # Not age based because loses potentially good individuals
        # just do selection again
        # In this case, sample without replacement. (selection was with replacement, so allowed duplicates)
        return self.selection(joinedPopulation, self.k, self.popul_size, fitness_scores, allow_duplicates=False)

    def distance(self, individual1, individual2):
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

    def get_single_fitness_shared(self, org_fitness, population, subpopulation, sub_popul_size, sub_popul_percent, i,
                                  n):
        # A single individual must be compared to all individuals in the subpopulation

        sharing_vals = [
            self.sharing_function(
                self.distance(population[i], subpopulation[j]) if not (i == j) else 0,  # if i == j then distance is 0
                max_distance=n)
            for j in range(sub_popul_size)]

        sum = np.sum(sharing_vals)  # dependent on the subpopulation sizedfsafds
        # So to rescale, we divide by the subpopulation percent
        sum = sum / sub_popul_percent

        # add 1 to the sharing val (sum) for the remaining 90% of the population (as explained in `fitness_sharing` func due to subpopul not including all individuals)
        sum += 1 if i >= sub_popul_size else 0

        fitness_shared = org_fitness * sum
        return fitness_shared, sum  # the sum is the penalty for being close to another individual

    def fitness_sharing(self, fitnesses_org, population):
        """
        The fitness sharing function is used to punish individuals that are close to other individuals.
        However, the fitness sharing function is computationally expensive, so we only compute it for the first 10% of the population.
        In addition for this 10%, we only compare it to the first 10% of the population, again for computational reasons. This gives an estimate of the real fitness sharing value.

        The remaining 90% of the population receives a sharing val of fitness * avg sharing val multiplier.
        """
        popul_size = len(population)
        n = len(population[0])

        # randomly take 10% of the population to consider for sharing
        sub_popul_percent = 0.1  # problem w/ subpopul is that for specific individuals, distance is 0 and for its very large
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
            [self.get_single_fitness_shared(
                fitnesses_org[i],
                population,
                subpopulation,
                sub_popul_size,
                sub_popul_percent, i, n)
                for i in indices_steekproef])

        fitnesses_shared = fitnesses_shared_and_sum[:, 0]
        penalty_sums = fitnesses_shared_and_sum[:, 1]

        # of these the shared_fitness is not computed, and their shared_fitness is defined as `fitness * avg penalty sum`
        indices_avg = np.arange(popul_size - sub_popul_size)

        # the rest of the population receives a sharing val of fitness * avg sharing val multiplier
        avg_sum = np.mean(penalty_sums[indices_steekproef])
        fitnesses_shared[indices_avg] = fitnesses_org[indices_avg] * avg_sum

        return fitnesses_shared

    def fitness_sharing_slow(self, fitnesses_org, population):
        fitnesses = fitnesses_org.copy()
        popul_size = len(population)
        n = len(population[0])

        # randomly take 10% of the population to consider for sharing
        sub_popul_percent = 0.1  # problem w/ subpopul is that for specific individuals, distance is 0 and for its very large
        # so can be that indiv is not included in neighbourhood so sharing val is 0. needs to be included in neighbourhood!
        sub_popul_size = int(popul_size * sub_popul_percent)

        subpopulation = population[:sub_popul_size]  # take first 10% of population
        # for remaining 90% of population, add 1 to the sharing val

        for i in range(len(population)):

            sharing_vals = np.zeros(sub_popul_size, dtype=np.float64)

            for j in range(sub_popul_size):
                dist = self.distance(population[i], subpopulation[j])

                sharing_val = self.sharing_function(dist, max_distance=n)
                sharing_vals[j] += sharing_val

            sum = np.sum(sharing_vals)  # dependent on the subpopulation sizedfsafds
            # So to rescale, we divide by the subpopulation percent
            sum = sum / sub_popul_percent

            # add 1 to the sharing val (sum) for the remaining 90% of the population (as explained above due to subpopul not including all individuals)
            sum += 1 if i >= sub_popul_size else 0
            fitnesses[i] = fitnesses[i] * sum

        return fitnesses

    def sharing_function(self, d, max_distance):
        # sigma_share is based on the maximum distance between any two individuals in the population, which is n
        # so only punish a candidate solution if it has a neighbour that is 1% of the max distance away
        # with similarity = # edges in common
        # so if path is 750 cities, punish node if it has a neighbour w/ 7.5 edges in common
        # sigma_share = max_distance * 0.1
        # sigma_share = max_distance * 0.2  # half of max distance

        # = max neighbourhood distance
        # sigma_share++ increases the neighbourhood distance -> more individuals are punished
        # sigma_share-- decreases the neighbourhood distance -> less individuals are punished
        sigma_share = max_distance * 0.2

        # sigma_share = int(max_distance * 0.2) # start punishing when 750 * 0.2 = 150 edges in common

        # alpha++ increases the penalty for being close to another individual
        # alpha-- decreases the penalty for being close to another individual
        alpha = 1
        if d <= sigma_share:
            val = 1 - (d / sigma_share) ** alpha
            return val
        else:
            return 0

    def avg_dist_func(self, population):
        """
        Approximation of the average distance between individuals in the population
        """
        average_distance = np.mean(
            [self.distance(population[i], population[i + 1]) for i in range(len(population) - 1)])
        return average_distance


if __name__ == "__main__":
    n = 4

    # parent = np.array([1, 3, 0, 2]) #np.random.permutation(n)
    # parent = np.array([0, 3, 1, 2])  # np.random.permutation(n)

    print("*" * 20)
    parent = np.array([2, 1, 3])  # 0, 2, 1, 3 but 0 is implicit
    e = EvolAlgorithm(None)
    parent_cyclic = e.edge_table(parent, n)  # 2 3 1 0
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
    offspring = e.crossover(selected)
    print(offspring)

    print("*" * 20)
    print("Testing distance")

    print(e.distance(np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])))
    print(e.distance(np.array([1, 2, 3, 4]), np.array([1, 2, 4, 3])))

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
