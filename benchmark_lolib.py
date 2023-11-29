import numpy as np


class Benchmark:
    def __init__(self, filename):
        self.matrix = self.read_matrix_from_file(filename)

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

    # def normalize_distance_matrix(self, distanceMatrix):
    #     # normalize distance matrix to be between 0 and 1
    #     # it makes the w's smaller and thus less likely to overflow
    #     distanceMatrix = distanceMatrix / np.max(distanceMatrix)
    #     return distanceMatrix

    def compute_fitness(self, population):  # slow, but easy to understand
        # shape: (populationSize, numCities)
        # eg population: [[1,2,3,4,5],[1,2,3,4,5], ... ]

        fitnesses = []
        n = len(population[0])
        for indiv_idx in range(n):  # iterate over population
            individual = population[indiv_idx]
            fitness = 0

            for i in range(n - 1):
                for j in range(i + 1, n):
                    fitness += self.matrix[individual[i]][individual[j]]
                    # fitness += self.matrix[int(individual[i])][int(individual[j])]

            fitnesses.append(fitness)
        return np.array(fitnesses)


if __name__ == '__main__':
    be75eec = Benchmark("benchmarks/be75eec.mat")
