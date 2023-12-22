import numpy as np


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
