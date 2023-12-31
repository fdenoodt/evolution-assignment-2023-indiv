import numpy as np


from abstract_benchmark import AbstractBenchmark


#
#
# if __name__ == "__main__":
#     # test whether compute_fitness and compute_fitness_good give the same results
#     filename = "./tour750.csv"
#     benchmark = Benchmark(filename, normalize=False, maximise=False)
#
#     population = np.random.rand(1000, 50) * 50
#     population = population.astype(int)
#
#     fitnesses = benchmark.compute_fitness(population)
#     fitnesses_slow = benchmark.compute_fitness_slow(population)
#     fitness_explicit = benchmark.compute_fitness_explicit(population)
#
#     assert np.all(np.allclose(fitnesses, fitnesses_slow))
#     assert np.all(np.allclose(fitnesses, fitness_explicit))
#
#     # Solve
#     # permutation, distance = benchmark.meta_solve()
#     # # permutation, distance = benchmark.dp_solve()
#     # print(permutation)
#     # print(distance)
