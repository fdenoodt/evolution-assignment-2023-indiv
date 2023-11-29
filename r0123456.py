import reporter as Reporter
import numpy as np

from benchmark_tsp import Benchmark
from placket_luce import PlackettLuce
from utility import Utility


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
        benchmark = Benchmark(filename)

        # fitness function
        f = lambda indiv: benchmark.compute_fitness(np.array([indiv]))[0]

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


# if __name__ == '__main__':
    # distanceMatrix = np.array([[0, 1, 2, 3, 4],
    #                            [np.inf, 0, 1, 2, 3],  # 1 -> 0 has dist inf
    #                            [2, 1, 0, 1, 2],
    #                            [3, 2, 1, 0, 1],
    #                            [4, 3, 2, 1, 0]])
    #
    # individual = np.array([4, 0, 2, 1, 3])
    # population = np.array([individual])
    # b = compute_fitness(population, distanceMatrix)
