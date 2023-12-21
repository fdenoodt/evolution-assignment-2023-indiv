import random

from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from placket_luce import PlackettLuce

import numpy as np


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
        f = self.benchmark.compute_fitness
        maximize = self.benchmark.maximise

        keep_running_until_timeup = keep_running_until_timeup
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.benchmark)

        # Initialize population
        population = self.initialize_population(self.popul_size, n)
        ctr = 0
        while True:

            fitnesses = f(population)

            best_fitness, mean_fitness, sigma_best = score_tracker.update_scores(
                fitnesses, population, ctr, pdf=None,
                print_w=False)  # pdf, w is only applicable to PlackettLuce, not Evol

            if ctr % 10 == 0:
                average_distance = np.mean(
                    [self.distance(population[i], population[i + 1]) for i in range(len(population) - 1)])
                print(f"Average distance: {average_distance}")

            # Selection
            selected = self.selection(population, self.k, self.offspring_size, fitnesses)

            # Variation
            offspring = self.crossover(selected)
            self.mutation(offspring, self.mutation_rate)  # overwrites the offspring
            joined_popul = np.vstack((offspring, population))

            # Evaluation / elimination
            fitnesses = f(joined_popul)
            population = self.elimination(joined_popul, fitnesses)

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

        deleted_individuals = np.bool_(np.zeros(len(population)))  # default: false
        nb_cities = np.size(population, 1)
        selected = np.zeros((nb_individuals_to_select, nb_cities))
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
            ri = random.choices(range(popul_size), k=k)
            min = np.argmin(fitness_scores[ri])  # take the single best
            best_indiv_idx = ri[min]
            selected[ii, :] = population[best_indiv_idx, :]
        return selected  # this may contain duplicates

    def edge_table(self, parent, nb_cities):
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

    # do edge crossover
    def crossover(self, selected):
        offspring = self.edge_crossover(selected)
        return offspring

    def single_cross_over_step(self, idx, curr_elt, father_ciclic, mother_ciclic, deleted_cities):
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
                length_city1 = self.get_list_length(father_ciclic, mother_ciclic, next_city1, deleted_cities)
                length_city2 = self.get_list_length(father_ciclic, mother_ciclic, next_city2, deleted_cities)

                if length_city1 < length_city2:
                    curr_elt = next_city1
                elif length_city1 > length_city2:
                    curr_elt = next_city2
                else:
                    curr_elt = next_city1 if random.randint(0, 1) == 0 else next_city2

            else:  # both are deleted so pick random
                available_cities = np.where(deleted_cities == False)
                available_cities = available_cities[0]  # cause its a tuple

                assert len(available_cities) > 0

                curr_elt = random.choice(available_cities)

        # • Ties are split at random
        #   Just pick from the father, if common then its also part of mother.
        #   There is no randomness, but it is not needed I think since father and mother are random

        # city_points_to1 = father[curr_elt]
        # city_points_to2 = mother[curr_elt]
        # curr_elt = city_points_to1 if used_cities[city_points_to1] == 0 else city_points_to2

        # 6. In the case of reaching an empty list, the other end of the offspring is
        # examined for extension; otherwise a new element is chosen at random

        return curr_elt

    def edge_crossover(self, selected):
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
            father_ciclic = self.edge_table(father, nb_cities)  # implicit 0 is included
            mother_ciclic = self.edge_table(mother, nb_cities)

            deleted_cities = np.zeros(nb_cities, dtype=bool)  # default: false
            offspring = np.zeros(nb_cities, dtype=int)  # implicit 0 is included

            # 2. pick an initial elt at rnd and put it in the offspring
            city = random.randint(0, nb_cities - 1)
            curr_elt = city

            # 3. Set the variable current element = entry
            offspring[0] = curr_elt

            for idx in range(nb_cities - 1):
                curr_elt = self.single_cross_over_step(idx, curr_elt, father_ciclic, mother_ciclic, deleted_cities)
                offspring[idx + 1] = curr_elt

            # offsprings now also contain the 0 somewhere in the middle
            # so we need to shift it to the start and remove the final element (which is 0)
            zero_idx = np.where(offspring == 0)[0][0]
            offspring = np.roll(offspring, shift=-zero_idx, axis=0)
            offspring = offspring[1:]

            offsprings[i, :] = offspring

        return offsprings

    def mutation_slow(self, offspring):  # slow
        # swap mutation
        # for each individual, swap 2 cities
        nb_cities = np.size(offspring, 1)
        for i in range(len(offspring)):
            individual = offspring[i]
            idx1 = random.randint(0, nb_cities - 1)
            idx2 = random.randint(0, nb_cities - 1)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def mutation(self, offspring_popul, mutation_rate):
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

    def elimination(self, joinedPopulation, fitness_scores):
        # Not age based because loses potentially good individuals
        # just do selection again
        # In this case, sample without replacement. (selection was with replacement, so allowed duplicates)
        return self.selection(joinedPopulation, self.k, self.popul_size, fitness_scores, allow_duplicates=False)

    def get_list_length(self, father_ciclic, mother_ciclic, curr_elt, deleted_cities):
        next_city_f = father_ciclic[curr_elt]
        next_city_m = mother_ciclic[curr_elt]
        count = 0
        if not (deleted_cities[next_city_f]):  # deleted -> 0
            count += 1
        if not (deleted_cities[next_city_m]) and next_city_m != next_city_f:  # deleted -> 0
            count += 1
        return count

    def distance(self, individual1, individual2):
        # individual1 and individual2 are in adjacency representation
        # so we need to convert to cycle representation, then can easily calculate distance by counting edges
        n = len(individual1) + 1
        indiv1_cyclic = self.edge_table(individual1, n)  # eg: 1 2 3 4 --> 1 2 3 4 0
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
