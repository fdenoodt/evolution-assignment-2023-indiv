import reporter as Reporter
import numpy as np

from placket_luce import PlackettLuce
from utility import Utility


def replace_inf_with_large_val(distanceMatrix):
    # replace inf with largest non inf value * max number of cities
    # just max is not enough, needs to make sure that worst possible path is still better than a single inf
    largest_value = np.max(distanceMatrix[distanceMatrix != np.inf]) * len(distanceMatrix)
    distanceMatrix = np.where(distanceMatrix == np.inf,
                              largest_value, distanceMatrix)
    # faster for the start, finds existing solutions quicker but in long run not that much impact

    print("largest non inf val: ", largest_value)
    return distanceMatrix


def normalize_distance_matrix(distanceMatrix):
    # normalize distance matrix to be between 0 and 1
    # it makes the w's smaller and thus less likely to overflow
    distanceMatrix = distanceMatrix / np.max(distanceMatrix)
    return distanceMatrix


class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.num_cities = 50
        self.keep_running_until_timeup = True
        self.numIters = 1000

        self.lr = 0.1
        self.nb_samples_lambda = 10

        self.utility = Utility(self.reporter, self.keep_running_until_timeup, self.numIters)
        self.pl = PlackettLuce(self.num_cities)

    def initialize_population(self, population_size, num_cities):
        population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])
        return population

    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        distanceMatrix = replace_inf_with_large_val(distanceMatrix)
        distanceMatrix = normalize_distance_matrix(distanceMatrix)

        # fitness function
        f = lambda indiv: compute_fitness(np.array([indiv]), distanceMatrix)[0]


        self.optimize_plackett_luce(f, self.pl.U_identity, self.lr, self.nb_samples_lambda)

    def optimize_plackett_luce(self, fitness_function, U_trans_function, lr, nb_samples_lambda):
        w_log = np.zeros(self.num_cities)  # w is w_tilde
        sigma_best = np.zeros(self.num_cities)  # the best permutation so far
        best_fitness = np.inf

        ctr = 0
        while True:
            # sample from plackett luce
            delta_w_log_ps = np.zeros((nb_samples_lambda, self.num_cities))
            sigmas = np.zeros((nb_samples_lambda, self.num_cities), dtype=int)
            fitnesses = np.zeros(nb_samples_lambda)

            for i in range(nb_samples_lambda):
                # sample sigma_i from Plackett luce
                sigmas[i] = self.pl.sample_permutation(np.exp(w_log))
                fitnesses[i] = fitness_function(sigmas[i])

                delta_w_log_ps[i] = self.pl.calc_w_log_p(w_log, sigmas[i])  # returns a vector

                if fitnesses[i] < best_fitness:
                    best_fitness = fitnesses[i]
                    sigma_best = sigmas[i]

            delta_w_log_F = self.pl.calc_w_log_F(w_log, fitnesses,
                                              delta_w_log_ps, U_trans_function, nb_samples_lambda)
            w_log = w_log - (lr * delta_w_log_F)  # "+" for maximization, "-" for minimization

            avg_fitness = np.average(fitnesses)
            print(f"best fitness: {best_fitness}, avg fitness: {avg_fitness / nb_samples_lambda}")
            # self.print_array(np.exp(w_log), ctr, frequency=10)
            # self.print_array(delta_w_log_F, ctr, frequency=10)
            # self.print_array_2d(delta_w_log_ps, ctr, frequency=10)

            ctr += 1
            # TODO
            # if numerical problems occurred:
            #   w = almost degenerate distr with mode at sigma_best

            if self.utility.is_done(ctr):
                break

        return best_fitness, sigma_best


def compute_fitness(population, distanceMatrix):  # slow, but easy to understand
    # shape: (populationSize, numCities)
    # eg population: [[1,2,3,4,5],[1,2,3,4,5], ... ]

    fitnesses = []
    for i in range(len(population)):
        individual = population[i]
        fitness = 0
        for j in range(len(individual)):
            city = individual[j]
            nextCity = individual[(j + 1) % len(individual)]
            fitness += distanceMatrix[int(city)][int(nextCity)]

        fitnesses.append(fitness)
    return np.array(fitnesses)


def compute_fitness_good(population, distanceMatrix):  # faster, generated with copilot but we understand it!

    # assert population doesn't contain cities that are floats (sanity check, can be removed later)
    assert np.all(np.equal(np.mod(population, 1), 0))

    # the faster way
    fitnesses = np.array([
        sum([distanceMatrix[int(city)][int(nextCity)] for city, nextCity in zip(individual, np.roll(individual, -1))])
        for individual in population])

    # returns: (populationSize, 1)
    # eg: [100,200, ... ]
    return fitnesses


if __name__ == '__main__':
    distanceMatrix = np.array([[0, 1, 2, 3, 4],
                               [np.inf, 0, 1, 2, 3],  # 1 -> 0 has dist inf
                               [2, 1, 0, 1, 2],
                               [3, 2, 1, 0, 1],
                               [4, 3, 2, 1, 0]])

    individual = np.array([4, 0, 2, 1, 3])
    population = np.array([individual])
    b = compute_fitness(population, distanceMatrix)
