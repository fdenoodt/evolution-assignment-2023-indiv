import numpy as np


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


if __name__ == "__main__":
    # fixed seed
    np.random.seed(123456)
    #
    # print("*" * 20)
    # print("Testing swap mutation")
    #
    n = 10
    # popul_size = 5
    # mutation_rate = 1
    # popul = np.array([np.arange(n) for _ in range(popul_size)])
    # print("popul:")
    # print(popul)
    # Variation.swap_mutation(popul, mutation_rate)
    # print("after mutate:")
    # print(popul)
    #
    # print("*" * 20)
    # print("Testing inversion mutation")
    # popul = np.array([np.arange(n) for _ in range(popul_size)])
    # print("popul:")
    # print(popul)
    # Variation.inversion_mutation(popul, mutation_rate)
    # print("after mutate:")
    # print(popul)
    #
    # print("*" * 20)
    # print("Testing scramble mutation")
    # popul = np.array([np.arange(n) for _ in range(popul_size)])
    # print("popul:")
    # print(popul)
    # Variation.scramble_mutation(popul, mutation_rate)
    # print("after mutate:")
    # print(popul)

    print("*" * 20)
    print("Testing order crossover")
    # popul = np.array([np.arange(1, n) for _ in range(2)])
    popul = np.array([np.arange(1, n), np.random.permutation(n - 1) + 1])
    print("popul before:")
    print(popul)
    popul = Variation.order_crossover(popul)
    print("popul after:")
    print(popul)
