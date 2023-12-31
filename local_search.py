import numpy as np

from benchmark_tsp import Benchmark





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
