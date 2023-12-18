import numpy as np
import numpy as np

from abstract_benchmark import AbstractBenchmark


class Benchmark(AbstractBenchmark):
    def __init__(self, filename, normalize, maximise):
        _matrix = self.read_matrix_from_file(filename)
        super().__init__(_matrix, normalize, maximise)

    def read_dimensions_from_file(self, file_path):
        with open(file_path, 'r') as file:
            _ = file.readline()  # skip first line
            line2 = file.readline()  # eg: "50\n"
            return int(line2)

    def read_matrix_from_file(self, file_path):
        dim = self.read_dimensions_from_file(file_path)

        data = np.loadtxt(file_path, skiprows=2)
        rows, cols = data.shape  # eg: 250, 10 -> so must reshape to 50, 50
        assert rows * cols == dim * dim

        matrix = data.reshape((dim, dim))

        # temporarily set dim to 10 to test
        matrix = matrix[:10, :10]


        return matrix

    def compute_fitness(self, population):
        return Benchmark.compute_fitness_static_fastest(self.matrix, population)

    @staticmethod
    def compute_fitness_static_fastest(matrix, population):
        num_cities = population.shape[1]

        # Create indices for all pairs of cities
        indices_i, indices_j = np.triu_indices(num_cities, k=1)

        # Use advanced indexing to calculate fitness for all pairs of cities simultaneously
        fitnesses = np.sum(matrix[population[:, indices_i], population[:, indices_j]], axis=1)
        return fitnesses.astype(np.float32)


if __name__ == '__main__':
    be75eec = Benchmark("benchmarks/be75eec.mat")
    population = np.random.randint(0, 50, size=(100, 50))
    # time the function's performance
    import timeit

    print(timeit.timeit(lambda: be75eec.compute_fitness(population), number=10))
    print(timeit.timeit(lambda: be75eec.compute_fitness(population), number=10))
