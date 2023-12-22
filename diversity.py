import numpy as np

from variation import Variation


class Diversity:
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
                                  n):
        # A single individual must be compared to all individuals in the subpopulation

        sharing_vals = [
            Diversity.sharing_function(
                Diversity.distance(population[i], subpopulation[j]) if not (i == j) else 0,
                # if i == j then distance is 0
                max_distance=n)
            for j in range(sub_popul_size)]

        sum = np.sum(sharing_vals)  # dependent on the subpopulation sizedfsafds
        # So to rescale, we divide by the subpopulation percent
        sum = sum / sub_popul_percent

        # add 1 to the sharing val (sum) for the remaining 90% of the population (as explained in `fitness_sharing` func due to subpopul not including all individuals)
        sum += 1 if i >= sub_popul_size else 0

        fitness_shared = org_fitness * sum
        return fitness_shared, sum  # the sum is the penalty for being close to another individual

    @staticmethod
    def fitness_sharing(fitnesses_org, population):
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
            [Diversity.get_single_fitness_shared(
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

    @staticmethod
    def fitness_sharing_slow(fitnesses_org, population):
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
                dist = Diversity.distance(population[i], subpopulation[j])

                sharing_val = Diversity.sharing_function(dist, max_distance=n)
                sharing_vals[j] += sharing_val

            sum = np.sum(sharing_vals)  # dependent on the subpopulation sizedfsafds
            # So to rescale, we divide by the subpopulation percent
            sum = sum / sub_popul_percent

            # add 1 to the sharing val (sum) for the remaining 90% of the population (as explained above due to subpopul not including all individuals)
            sum += 1 if i >= sub_popul_size else 0
            fitnesses[i] = fitnesses[i] * sum

        return fitnesses

    @staticmethod
    def sharing_function(d, max_distance):
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

    @staticmethod
    def avg_dist_func(population):
        """
        Approximation of the average distance between individuals in the population
        """
        average_distance = np.mean(
            [Diversity.distance(population[i], population[i + 1]) for i in range(len(population) - 1)])
        return average_distance
