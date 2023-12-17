import reporter as Reporter
from placket_luce import PlackettLuce, PdfRepresentation
from utility import Utility

import numpy as np


class r0123456:
    def __init__(self, lr, nb_samples_lambda, numIters, U, benchmark):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.keep_running_until_timeup = True
        self.numIters = numIters

        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda

        self.utility = Utility(self.reporter, self.keep_running_until_timeup, self.numIters)
        self.pl = PlackettLuce(U, benchmark)

    def optimize(self, pdf):
        n = self.pl.benchmark.permutation_size()
        f = self.pl.benchmark.compute_fitness
        self.optimize_plackett_luce(f, self.lr, self.nb_samples_lambda, n, pdf)

    def optimize_plackett_luce(self, fitness_func, lr, nb_samples_lambda, n, pdf):

        # specify data types for numba
        sigma_best = np.zeros(n, dtype=np.int64)
        best_fitness = 0

        ctr = 0
        while True:

            # Sample sigma_i from Plackett luce
            sigmas = pdf.sample_permutations(nb_samples_lambda)
            fitnesses = fitness_func(sigmas)

            delta_w_log_ps = pdf.calc_gradients(sigmas)

            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_fitness:
                best_fitness = fitnesses[best_idx]
                sigma_best = sigmas[best_idx]

            delta_w_log_F = PlackettLuce.calc_w_log_F(
                self.pl.U, fitnesses, delta_w_log_ps, nb_samples_lambda)

            pdf.update_w_log(delta_w_log_F, lr)

            avg_fitness = np.mean(fitnesses)

            self.utility.print_score(ctr, best_fitness, avg_fitness, nb_samples_lambda)
            # self.utility.print_array((w_log), ctr, frequency=10)
            # self.utility.print_array(np.exp(w_log), ctr, frequency=10)
            # self.utility.print_array(delta_w_log_F, ctr, frequency=10)
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
