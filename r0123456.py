import reporter as Reporter
import numpy as np
import random


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

    def initialize_population(self, population_size, num_cities):
        population = np.array([np.random.permutation(num_cities) for _ in range(population_size)])
        return population

    def U_identity(self, x):
        return x

    def U_normalize(self, x, xs):
        return x / np.sum(xs)

    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        distanceMatrix = replace_inf_with_large_val(distanceMatrix)
        distanceMatrix = normalize_distance_matrix(distanceMatrix)

        nb_samples_lambda = 10

        # fitness function
        f = lambda indiv: compute_fitness(np.array([indiv]), distanceMatrix)[0]

        lr = 0.1

        self.optimize_plackett_luce(f, self.U_identity, lr, nb_samples_lambda)

    def print_array(self, arr, ctr, frequency=10):
        if ctr % frequency == 0:
            w_exp = np.array([f'{a:.2f}' for a in arr])
            print(w_exp)

    def print_array_2d(self, arr, ctr, frequency=10):
        if ctr % frequency == 0:
            for i in range(len(arr)):
                temp = np.array([f'{a:.2f}' for a in arr[i]])
                print(" ".join(temp))
            print()
            print()
            print()

    def is_done(self, i, meanObjective=0, bestObjective=0, bestSolution=np.array([])):
        timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
        i += 1
        time_over = (timeLeft < 0 and self.keep_running_until_timeup)
        iters_over = (not (self.keep_running_until_timeup) and i > self.numIters)
        return (time_over or iters_over)

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
                sigmas[i] = self.sample_permutation(np.exp(w_log))
                fitnesses[i] = fitness_function(sigmas[i])

                delta_w_log_ps[i] = self.calc_w_log_p(w_log, sigmas[i])  # returns a vector

                if fitnesses[i] < best_fitness:
                    best_fitness = fitnesses[i]
                    sigma_best = sigmas[i]

            delta_w_log_F = self.calc_w_log_F(w_log, fitnesses,
                                              delta_w_log_ps, U_trans_function, nb_samples_lambda)
            w_log = w_log - (lr * delta_w_log_F)  # "+" for maximization, "-" for minimization

            # avg_fitness = np.mean(fitnesses)
            # print(f"best fitness: {best_fitness}, avg fitness: {avg_fitness / nb_samples_lambda}")
            self.print_array(np.exp(w_log), ctr, frequency=10)
            # self.print_array(delta_w_log_F, ctr, frequency=10)
            # self.print_array_2d(delta_w_log_ps, ctr, frequency=10)

            ctr += 1
            # TODO
            # if numerical problems occurred:
            #   w = almost degenerate distr with mode at sigma_best

            if self.is_done(ctr):
                break

        return best_fitness, sigma_best

    def sample_permutation(self, w):
        n = self.num_cities
        sigma = np.zeros(n, dtype=int)
        used_nodes = np.zeros(n, dtype=bool)

        for i in range(n):
            node = self.sample_node(w, used_nodes)  # should return one city
            sigma[i] = node
            used_nodes[node] = True

        return sigma

    # multiple samples
    # def sample_permutation(self, w, size=1):
    #     n = self.num_cities
    #     samples = np.zeros((size, n), dtype=int)
    #
    #     for i in range(size):
    #         sigma = np.zeros(n, dtype=int)
    #         used_nodes = np.zeros(n, dtype=bool)
    #
    #         for j in range(n):
    #             node = self.sample_node(w, used_nodes)  # should return one city
    #             sigma[j] = node
    #             used_nodes[node] = True
    #
    #         samples[i] = sigma
    #
    #     return samples

    def sample_node(self, w, used_nodes):
        # compute probabilities: its the values in w, except its zero for used nodes
        probabilities = w.copy()
        probabilities[used_nodes] = 0
        probabilities /= np.sum(
            probabilities)  # TODO: COULD BE THAT W'S ARE STILL LOGS SO NEED TO EXP THEM OR VICE VERSA

        # check if probabilities contain NaNs
        if np.isnan(probabilities).any():
            assert False

        # sample from probabilities
        node = np.random.choice(self.num_cities, p=probabilities)

        return node

    def calc_w_log_p_partial(self, w_log, sigma, i):
        n = len(sigma)

        intermediate_result = 0  # calc \Sigma (1/sum)
        for k in range(i):
            sum = 0  # calc sum in denominator
            for j in range(k, n):
                sum += np.exp(w_log[sigma[j]])

            intermediate_result += 1 / sum

        return 1 - np.exp(w_log[sigma[i]]) * intermediate_result

    def calc_w_log_p(self, w_log, sigma):
        # Calculates all partial derivatives for a sample sigma
        n = len(sigma)
        gradient = np.zeros_like(w_log)

        for i in range(n):
            gradient[sigma[i]] = self.calc_w_log_p_partial(w_log, sigma, i)

        return gradient


    def calc_w_log_F(self, w_log, fitnesses, delta_w_log_ps, U, nb_samples_lambda, ):
        gradient = np.zeros_like(w_log)

        for i in range(nb_samples_lambda):
            f_val = U(fitnesses[i])  # scalar
            gradient += f_val * delta_w_log_ps[i]  # scalar * vector

        gradient /= nb_samples_lambda

        return gradient


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
