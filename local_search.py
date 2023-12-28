import numpy as np

from benchmark_tsp import Benchmark


class LocalSearch:

    @staticmethod
    def insert_random_node(population, d, nb_nodes_to_insert_percent=0.1):
        nb_nodes_to_insert = int(np.size(population, 1) * nb_nodes_to_insert_percent)
        print(f"inserting {nb_nodes_to_insert} nodes")

        if nb_nodes_to_insert == 0:
            return population

        # verify perm_len: otherwise we would be inserting the same node
        max_perm_len = np.size(population, 1)
        assert nb_nodes_to_insert <= max_perm_len

        population = population.copy()
        nb_cities = np.size(population, 1)

        for indiv_idx, indivd in enumerate(population):
            # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

            # pick two random nodes
            # a_idx = np.random.randint(0, nb_cities - 1)
            # b_idx = np.random.randint(0, nb_cities - 1)
            # if a_idx == b_idx:
            #     continue

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
                    print(f"inserting node {b} after node {a} in individual {indiv_idx}")
                    # insert node b after node a and shift all other nodes
                    indivd = np.insert(indivd, a_idx + 1, indivd[b_idx])
                    if a_idx < b_idx:
                        indivd = np.delete(indivd, b_idx + 1)
                    else:
                        indivd = np.delete(indivd, b_idx)

                    population[indiv_idx] = indivd

        return population

    # @staticmethod
    # def reverse_two_nodes(population, d, nb_nodes_to_reverse_percent=0.1):
    #     nb_nodes_to_reverse = int(np.size(population, 1) * nb_nodes_to_reverse_percent)
    #     print(f"reversing {nb_nodes_to_reverse} nodes")
    #
    #     if nb_nodes_to_reverse == 0:
    #         return population
    #
    #     # verify perm_len: otherwise we would be inserting the same node
    #     max_perm_len = np.size(population, 1)
    #     assert nb_nodes_to_reverse <= max_perm_len
    #
    #     population = population.copy()
    #     nb_cities = np.size(population, 1)
    #
    #     for indiv_idx, indivd in enumerate(population):
    #         # pick `nb_nodes_to_insert` random nodes without replacement
    #         rnd_a_indices = np.random.choice(nb_cities, nb_nodes_to_reverse, replace=False)
    #         rnd_b_indices = np.random.choice(nb_cities, nb_nodes_to_reverse, replace=False)
    #
    #         for a_idx, b_idx in zip(rnd_a_indices, rnd_b_indices):
    #             if a_idx == b_idx:
    #                 continue
    #
    #             # check if better performance
    #             a = indivd[a_idx]
    #             b = indivd[b_idx]
    #             a_next = indivd[(a_idx + 1) % nb_cities]
    #             b_next = indivd[(b_idx + 1) % nb_cities]
    #
    #             # ... a_prev -> a -> a_next -> ...
    #             # ... b_prev -> b -> b_next -> ...
    #             current_cost = d[a, a_next] + d[b, b_next]

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


if __name__ == "__main__":
    np.random.seed(123456)
    # test whether compute_fitness and compute_fitness_good give the same results
    filename = "./tour750.csv"
    benchmark = Benchmark(filename, normalize=False, maximise=False)

    num_cities = benchmark.permutation_size()
    population_size = 10

    # # random permutations
    # population1 = np.array([np.random.permutation(num_cities - 1) + 1 for _ in range(population_size)], dtype=int)
    # population2 = population1.copy()
    #
    # fitnesses_before = benchmark.compute_fitness(population1)
    # print(f"before: {np.average(fitnesses_before):_.4f}")
    #
    # # population2 = LocalSearch.two_opt_swap(population2, benchmark.matrix)
    # population2 = LocalSearch.two_opt(population2, benchmark.matrix, 5)
    # fitnesses_after = benchmark.compute_fitness(population2)
    # print(f"fast: {np.average(fitnesses_after):_.4f}")
    #
    # population1 = LocalSearch.two_opt_slow(population1, benchmark.matrix)
    # fitnesses_after = benchmark.compute_fitness(population1)
    # print(f"two: {np.average(fitnesses_after):_.4f}")
    # assert np.all(np.less_equal(fitnesses_after, fitnesses_before))

    # print("test indivd[i + 1:j + 1] = indivd[j:i:-1]")
    # indivd = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # i = 2
    # j = 5
    # indivd[i + 1:j + 1] = indivd[j:i:-1]
    # print(indivd)

    # print("test two-opt_swap")
    # indivd = np.array([1, 2, 3, 4, 5, 6, 7])
    # print(indivd)
    # indivd = LocalSearch.two_opt_swap(np.array([indivd]), benchmark.matrix[0:8, 0:8])
    # print(indivd)

    print("test insert random node")
    filename = "./tour50.csv"
    benchmark = Benchmark(filename, normalize=False, maximise=False)
    num_cities = benchmark.permutation_size()
    population_size = 1
    population = np.array([np.random.permutation(num_cities - 1) + 1 for _ in range(population_size)], dtype=int)

    print(population)
    population_optimiz = LocalSearch.insert_random_node(population, benchmark.matrix, nb_nodes_to_insert_percent=1)

    print(population_optimiz)
    fitnesses_before = benchmark.compute_fitness(population)
    fitnesses_after = benchmark.compute_fitness(population_optimiz)
    print(f"before: {np.average(fitnesses_before):_.4f}")
    print(f"after: {np.average(fitnesses_after):_.4f}")

    assert np.all(np.less_equal(fitnesses_after, fitnesses_before))
