import reporter as Reporter
from ScoreTracker import ScoreTracker
from placket_luce import PlackettLuce
from utility import Utility

import numpy as np


class r0123456:
    def __init__(self, lr, nb_samples_lambda, numIters, U, benchmark):
        self.reporter_name = self.__class__.__name__

        self.keep_running_until_timeup = False
        self.numIters = numIters

        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda

        self.pl = PlackettLuce(U, benchmark)

    def optimize(self, pdf):
        n = self.pl.benchmark.permutation_size()
        f = self.pl.benchmark.compute_fitness
        maximize = self.pl.benchmark.maximise
        reporter_name = self.reporter_name
        keep_running_until_timeup = self.keep_running_until_timeup
        numIters = self.numIters

        # stores best score + best sigma
        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name)
        self.optimize_plackett_luce(f, self.lr, self.nb_samples_lambda, n, pdf, maximize, score_tracker)

    def optimize_plackett_luce(self, fitness_func, lr, nb_samples_lambda, n, pdf, maximize, score_tracker):
        ctr = 0
        while True:
            # Sample sigma_i from Plackett luce
            sigmas = pdf.sample_permutations(nb_samples_lambda)
            fitnesses = fitness_func(sigmas)

            delta_w_log_ps = pdf.calc_gradients(sigmas)

            best_fitness, sigma_best = score_tracker.update_scores(fitnesses, sigmas, ctr, pdf, print_w=True)

            delta_w_log_F = PlackettLuce.calc_w_log_F(self.pl.U, fitnesses, delta_w_log_ps, nb_samples_lambda)

            pdf.update_w_log(delta_w_log_F, lr, maximize)

            # TODO
            # if numerical problems occurred:
            #   w = almost degenerate distr with mode at sigma_best

            ctr += 1
            if score_tracker.utility.is_done(ctr):
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
