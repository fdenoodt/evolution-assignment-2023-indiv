import numpy as np


class Utility:

    def __init__(self, reporter, keep_running_until_timeup, numIters):
        self.reporter = reporter
        self.keep_running_until_timeup = keep_running_until_timeup
        self.numIters = numIters

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
