import numpy as np

from benchmark_tsp import Benchmark


class LocalSearch:

    @staticmethod
    def two_opt(population, d):
        # https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
        population = population.copy()
        population = np.hstack((np.zeros((len(population), 1), dtype=int), population))
        nb_cities = np.size(population, 1)

        # since matrix is not symmetric, we need to check both directions
        for indiv_idx, indivd in enumerate(
                population):  # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

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
                        # reverse path from a_next to b
                        indivd[i + 1:j + 1] = indivd[j:i:-1]

        population = population[:, 1:]

        return population

    @staticmethod
    def two_opt_swap(population, d):
        population = population.copy()
        population = np.hstack((np.zeros((len(population), 1), dtype=int), population))
        nb_cities = np.size(population, 1)

        # since matrix is not symmetric, we need to check both directions
        for indiv_idx, indivd in enumerate(
                population):  # based on: https://dm865.github.io/assets/dm865-tsp-ls-handout.pdf

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

    # random permutations
    population1 = np.array([np.random.permutation(num_cities - 1) + 1 for _ in range(population_size)], dtype=int)
    population2 = population1.copy()

    fitnesses_before = benchmark.compute_fitness(population1)
    population1 = LocalSearch.two_opt(population1, benchmark.matrix)
    fitnesses_after = benchmark.compute_fitness(population1)

    population2 = LocalSearch.two_opt_swap(population2, benchmark.matrix)
    print(f"before: {np.average(fitnesses_before):_.4f}")
    print(f"two: {np.average(fitnesses_after):_.4f}")
    print(f"two swap: {np.average(fitnesses_before) - np.average(fitnesses_after):_.4f}")

    assert np.all(np.less_equal(fitnesses_after, fitnesses_before))
