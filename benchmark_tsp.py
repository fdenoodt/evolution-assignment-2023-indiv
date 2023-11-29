import numpy as np


class Benchmark:
    def __init__(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        distanceMatrix = self.replace_inf_with_large_val(distanceMatrix)
        self.distanceMatrix = self.normalize_distance_matrix(distanceMatrix)

    def permutation_size(self):
        return self.distanceMatrix.shape[0]

    def replace_inf_with_large_val(self, distanceMatrix):
        # replace inf with largest non inf value * max number of cities
        # just max is not enough, needs to make sure that worst possible path is still better than a single inf
        largest_value = np.max(distanceMatrix[distanceMatrix != np.inf]) * len(distanceMatrix)
        distanceMatrix = np.where(distanceMatrix == np.inf,
                                  largest_value, distanceMatrix)
        # faster for the start, finds existing solutions quicker but in long run not that much impact

        print("largest non inf val: ", largest_value)
        return distanceMatrix

    def normalize_distance_matrix(self, distanceMatrix):
        # normalize distance matrix to be between 0 and 1
        # it makes the w's smaller and thus less likely to overflow
        distanceMatrix = distanceMatrix / np.max(distanceMatrix)
        return distanceMatrix

    def compute_fitness(self, population):  # slow, but easy to understand
        # shape: (populationSize, numCities)
        # eg population: [[1,2,3,4,5],[1,2,3,4,5], ... ]

        fitnesses = []
        for i in range(len(population)):
            individual = population[i]
            fitness = 0
            for j in range(len(individual)):
                city = individual[j]
                nextCity = individual[(j + 1) % len(individual)]
                fitness += self.distanceMatrix[int(city)][int(nextCity)]

            fitnesses.append(fitness)
        return np.array(fitnesses)

    def compute_fitness_good(self, population, distanceMatrix):  # faster, generated with copilot but we understand it!

        # assert population doesn't contain cities that are floats (sanity check, can be removed later)
        assert np.all(np.equal(np.mod(population, 1), 0))

        # the faster way
        fitnesses = np.array([
            sum([distanceMatrix[int(city)][int(nextCity)] for city, nextCity in
                 zip(individual, np.roll(individual, -1))])
            for individual in population])

        # returns: (populationSize, 1)
        # eg: [100,200, ... ]
        return fitnesses
