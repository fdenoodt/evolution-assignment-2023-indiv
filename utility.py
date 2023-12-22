import numpy as np


class Utility:

    def __init__(self, reporter, keep_running_until_timeup, numIters):
        self.reporter = reporter
        self.keep_running_until_timeup = keep_running_until_timeup
        self.numIters = numIters

    def print_score(self, ctr, best_fitness, avg_fitness, frequency=10, avg_dist_func=None, fitnesses_shared=None):
        # avg_dist_func can only be used if fitnesses_shared is not None
        assert (avg_dist_func is None) or (fitnesses_shared is not None)

        if ctr % frequency == 0:
            # print(f"{ctr} \t best fitness: {best_fitness:_.4f}, avg fitness: {avg_fitness:_.4f}")

            if callable(avg_dist_func):
                average_distance = avg_dist_func()
                avg_fitness_shared = np.mean(fitnesses_shared)
                print(
                    f"{ctr} \t best fitness: {best_fitness:_.4f}, avg fitness: {avg_fitness:_.4f}, fit shared: {avg_fitness_shared:_.4f}, avg dist: {average_distance:_.4f}")
            else:
                print(f"{ctr} \t best fitness: {best_fitness:_.4f}, avg fitness: {avg_fitness:_.4f}")

    @staticmethod
    def print_mtx(mtx, ctr, frequency=10, sub_mtx=None):
        if sub_mtx is not None:
            mtx = mtx[:, :sub_mtx]

        if ctr % frequency == 0:
            for i in range(len(mtx)):
                temp = np.array([f'{a:.4f}' for a in mtx[i]])
                print(" ".join(temp))
            print()
            print()
            print()

    @staticmethod
    def print_array(arr, ctr, frequency=10):
        if ctr % frequency == 0:
            w_exp = np.array([f'{a:.4f}' for a in arr])
            print(w_exp)

    @staticmethod
    def print_array_2d(arr, ctr, frequency=10):
        if ctr % frequency == 0:
            for i in range(len(arr)):
                temp = np.array([f'{a:.2f}' for a in arr[i]])
                print(" ".join(temp))
            print()
            print()
            print()

    def is_done_and_report(self, i, meanObjective, bestObjective, bestSolution):
        timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
        time_over = (timeLeft < 0 and self.keep_running_until_timeup)
        iters_over = (not (self.keep_running_until_timeup) and i > self.numIters)
        return (time_over or iters_over)
