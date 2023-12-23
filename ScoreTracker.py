import numpy as np
import reporter as Reporter
from utility import Utility


class ScoreTracker:
    def __init__(self, n, maximize, keep_running_until_timeup, numIters, reporter_name, benchmark):
        self.maximize = maximize
        self.all_time_best_fitness = -np.inf if maximize else np.inf
        self.all_time_sigma_best = np.zeros(n, dtype=np.int64)
        reporter = Reporter.Reporter(reporter_name)
        self.utility = Utility(reporter, keep_running_until_timeup, numIters)
        self.benchmark = benchmark

    def update_scores(self, fitnesses, sigmas, ctr, fitnesses_shared=None, pdf=None, print_w=False, avg_dist_func=None,
                      island_identifier=None, print_score=True):

        fitnesses = self.benchmark.unnormalize_fitnesses(fitnesses)
        if fitnesses_shared is not None:
            fitnesses_shared = self.benchmark.unnormalize_fitnesses(fitnesses_shared)

        # code is clearer
        if self.maximize:
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            sigma_best = sigmas[best_idx]
            if best_fitness > self.all_time_best_fitness:
                self.all_time_best_fitness = best_fitness
                self.all_time_sigma_best = sigma_best
        else:
            best_idx = np.argmin(fitnesses)
            best_fitness = fitnesses[best_idx]
            sigma_best = sigmas[best_idx]
            if best_fitness < self.all_time_best_fitness:
                self.all_time_best_fitness = best_fitness
                self.all_time_sigma_best = sigma_best

        if print_w and pdf is not None:
            assert island_identifier is None

            w = np.exp(pdf.w_log)
            frequency = 100
            if len(w.shape) == 2:  # if w_log is square matrix:
                Utility.print_mtx(w, ctr, frequency, sub_mtx=10)
            elif len(w.shape) == 1:  # if w_log is 1d array:
                Utility.print_array(w, ctr, frequency)
            else:
                raise Exception("w_log has unsupported shape")

        avg_fitness = np.mean(fitnesses)

        if print_score:
            Utility.print_score(ctr, best_fitness, avg_fitness, 10, avg_dist_func, fitnesses_shared,
                                     island_identifier)
        return best_fitness, avg_fitness, sigma_best
