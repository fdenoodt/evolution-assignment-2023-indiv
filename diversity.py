import numpy as np

from utility import Utility
from variation import Variation


class Island:
    def __init__(self, identifier, f, popul_size, n):
        self.identifier = identifier
        self.f = f
        self.popul_size = popul_size
        self.population = Island.initialize_population(self.popul_size, n)

    @staticmethod
    def initialize_population(population_size, num_cities):
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

    @staticmethod
    def run_epochs(nb_epochs, islands, selection, elimination, mutation, score_tracker, ctr):
        best_sigma, last_fitnesses_shared = None, []

        # contains all fitnesses of a single island (for all epochs)
        best_fitnesses = np.zeros(nb_epochs, dtype=np.float64)
        mean_fitnesses = np.zeros(nb_epochs, dtype=np.float64)

        done = False
        nb_islands = len(islands)
        for idx, island in enumerate(islands):
            for epoch in range(nb_epochs):
                # overwrites best_fitness, mean_fitness, sigma_best, but that's ok to me
                best_fitnesses[epoch], mean_fitnesses[epoch], best_sigma, last_fitnesses_shared = island.step(
                    selection, elimination, mutation, score_tracker, epoch + ctr)

                if epoch == nb_epochs - 1:  # only print results for last epoch of each island
                    Utility.print_score(epoch + ctr, best_fitnesses[epoch], np.mean(mean_fitnesses), 1,
                                        avg_dist_func=lambda: FitnessSharing.avg_dist_func(island.population),
                                        fitnesses_shared=np.mean(last_fitnesses_shared),
                                        island_identifier=island.identifier)

                if idx == nb_islands - 1:  # only store results for last island
                    if score_tracker.utility.is_done_and_report(
                            ctr + epoch, mean_fitnesses[epoch], best_fitnesses[epoch], best_sigma):
                        done = True
                        break
        print()

        return done

    def step(self, selection, elimination, mutation, score_tracker, ctr):
        fitnesses_not_scaled = self.f(self.population)  # before fitness sharing

        # Fitness sharing (MUST COME AFTER score_tracker.update_scores)
        fitnesses = FitnessSharing.fitness_sharing(fitnesses_not_scaled, self.population)

        # Update scores
        best_fitness, mean_fitness, best_sigma = score_tracker.update_scores(
            fitnesses_not_scaled, self.population, ctr,
            fitnesses_shared=fitnesses,
            pdf=None, print_w=False,  # pdf, w is only applicable to PlackettLuce, not Evol
            # only applicable to Evol, not PlackettLuce
            avg_dist_func=lambda: FitnessSharing.avg_dist_func(self.population),
            island_identifier=self.identifier,
            print_score=False
        )

        # Selection
        selected = selection(self.population, fitnesses)

        # Variation
        offspring = Variation.crossover(selected)
        mutation(offspring)

        joined_popul = np.vstack((offspring, self.population))

        # Evaluation / elimination
        fitnesses = self.f(joined_popul)
        self.population = elimination(joined_popul, fitnesses)

        # shuffle popul in place, required because other functions such
        # Diversity.fitness_sharing uses the first 10% of the population assuming it is random
        np.random.shuffle(self.population)

        # sanity check
        # for i in range(len(population)):
        #     assert len(population[i]) == len(set(population[i])) == n - 1

        return best_fitness, mean_fitness, best_sigma, fitnesses

    @staticmethod
    def migrate(islands, popul_size, percentage=0.1):
        print("Migrating...")
        # 10% of the population migrates to the next island
        assert len(islands) > 1
        migrants = islands[-1].population[:int(popul_size * percentage)]
        for idx, island in enumerate(islands):
            migrants = island._migrate(migrants)

    def _migrate(self, other_island_migrants):
        nb_migrants = len(other_island_migrants)
        our_migrants = self.population[:nb_migrants].copy()  # take first nb_migrants, already shuffled
        self.population[:nb_migrants] = other_island_migrants

        # np.random.shuffle(self.population)

        return our_migrants

        # Function Entirely generated via Copilot

        # migrate 10% of the population
        # nb_migrants = int(self.popul_size * 0.1)
        # indices_migrants = np.random.choice(self.popul_size, nb_migrants, replace=False)
        #
        # # migrate the selected individuals
        # self.population[indices_migrants] = other_island.population[indices_migrants]


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
                                  n):
        # A single individual must be compared to all individuals in the subpopulation

        sharing_vals = [
            FitnessSharing.sharing_function(
                FitnessSharing.distance(population[i], subpopulation[j]) if not (i == j) else 0,
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
            [FitnessSharing.get_single_fitness_shared(
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
                dist = FitnessSharing.distance(population[i], subpopulation[j])

                sharing_val = FitnessSharing.sharing_function(dist, max_distance=n)
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
            [FitnessSharing.distance(population[i], population[i + 1]) for i in range(len(population) - 1)])
        return average_distance


if __name__ == "__main__":
    print("*" * 20)
    print("Test migration")
    n = 8
    popul_size = 4
    islands = [Island(i, None, popul_size, n) for i in range(3)]
    print("Before migration")
    print(islands[0].population)
    print(islands[1].population)
    print(islands[2].population)
    Island.migrate(islands, popul_size, percentage=0.5)
    print("After migration")
    print(islands[0].population)
    print(islands[1].population)
    print(islands[2].population)
