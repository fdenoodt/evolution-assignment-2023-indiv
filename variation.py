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
    def mutation_slow(offspring):  # slow
        # swap mutation
        # for each individual, swap 2 cities
        nb_cities = np.size(offspring, 1)
        for i in range(len(offspring)):
            individual = offspring[i]
            # idx1 = random.randint(0, nb_cities - 1)
            # idx2 = random.randint(0, nb_cities - 1)
            idx1 = np.random.randint(0, nb_cities)  # np.random.randint upperboun is exclusive
            idx2 = np.random.randint(0, nb_cities)  # np.random.randint upperboun is exclusive
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    @staticmethod
    def mutation(offspring_popul, mutation_rate):
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
