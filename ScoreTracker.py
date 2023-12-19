import numpy as np
import reporter as Reporter
from utility import Utility


class ScoreTracker:
    def __init__(self, n, maximize, keep_running_until_timeup, numIters, reporter_name):
        self.maximize = maximize
        self.best_fitness = -np.inf if maximize else np.inf
        self.sigma_best = np.zeros(n, dtype=np.int64)
        reporter = Reporter.Reporter(reporter_name)
        self.utility = Utility(reporter, keep_running_until_timeup, numIters)

    def update_scores(self, fitnesses, sigmas, ctr, pdf, print_mtx=False):
        # code is clearer
        if self.maximize:
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.sigma_best = sigmas[best_idx]
        else:
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.sigma_best = sigmas[best_idx]

        avg_fitness = np.mean(fitnesses)

        self.utility.print_score(ctr, self.best_fitness, avg_fitness, 10)

        if print_mtx:
            w = np.exp(pdf.w_log)
            frequency = 10
            if len(w.shape) == 2:  # if w_log is square matrix:
                Utility.print_mtx(w, ctr, frequency, sub_mtx=10)
            elif len(w.shape) == 1:  # if w_log is 1d array:
                Utility.print_array(w, ctr, frequency)
            else:
                raise Exception("w_log has unsupported shape")

        return self.best_fitness, self.sigma_best
