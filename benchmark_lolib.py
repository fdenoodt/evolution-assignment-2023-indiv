import numpy as np

from numba import int32, jit


class Benchmark:
    def __init__(self, filename, normalize=False):
        self.matrix = self.read_matrix_from_file(filename)
        self.normalizing_constant = 1
        if normalize:
            self.matrix, self.normalizing_constant = self.normalize_matrix(self.matrix)

    def normalize_matrix(self, matrix):
        # normalize distance matrix to be between 0 and 1
        # it makes the w's smaller and thus less likely to overflow
        constant = np.max(matrix)
        distanceMatrix = matrix / constant
        return distanceMatrix, constant

    def permutation_size(self):
        return self.matrix.shape[0]

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
        return matrix

    def compute_fitness(self, population):
        return Benchmark.compute_fitness_static(self.matrix, population)

    @staticmethod
    @jit(nopython=True)
    def compute_fitness_static(matrix, population):
        # shape: (populationSize, numCities)
        # eg population: [[1,2,3,4,5],[1,2,3,4,5], ... ]

        fitnesses = []
        popul_size = len(population)
        n = len(population[0])
        for indiv_idx in range(popul_size):  # iterate over population
            individual = population[indiv_idx]
            fitness = 0

            for i in range(n - 1):
                for j in range(i + 1, n):
                    fitness += matrix[individual[i]][individual[j]]
                    # fitness += self.matrix[int(individual[i])][int(individual[j])]

            fitnesses.append(fitness)

        # was initially a maximization problem, so we need to negate the fitnesses
        return np.array(fitnesses)


if __name__ == '__main__':
    be75eec = Benchmark("benchmarks/be75eec.mat")
    population = np.random.randint(0, 50, size=(100, 50))
    # time the function's performance
    import timeit

    print(timeit.timeit(lambda: be75eec.compute_fitness(population), number=1000))
    print(timeit.timeit(lambda: be75eec.compute_fitness(population), number=1000))
