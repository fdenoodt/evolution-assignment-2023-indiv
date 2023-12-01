import reporter as Reporter
from placket_luce import PlackettLuce
from utility import Utility

import torch
import torch.nn as nn
import torch.optim as optim


class r0123456:
    def __init__(self, lr, nb_samples_lambda, numIters, U):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.keep_running_until_timeup = True
        self.numIters = numIters

        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda

        self.utility = Utility(self.reporter, self.keep_running_until_timeup, self.numIters)
        self.pl = PlackettLuce(U)

    def optimize(self, benchmark):
        n = benchmark.permutation_size()

        # fitness function
        f = lambda indiv: (benchmark.compute_fitness(torch.unsqueeze(indiv.clone().detach(), 0))[0])
        self.optimize_plackett_luce(f, self.lr, self.nb_samples_lambda, n)

    def test_loss(self, w_log):
        # mse with ones vector
        loss = nn.MSELoss()
        target = torch.ones_like(w_log)
        return loss(w_log, target)


    def optimize_plackett_luce(self, fitness_function, lr, nb_samples_lambda, n):
        w_log = torch.zeros(n)  # w is w_tilde
        w_log.requires_grad = True

        sigma_best = torch.zeros(n)  # the best permutation so far
        best_fitness = torch.inf

        ctr = 0
        while True:
            w_log.grad = None
            loss = self.test_loss(w_log)
            loss.backward()

            with torch.no_grad():
                w_log -= lr * w_log.grad
                # w_log = torch.clamp(w_log, min=0)

            print(w_log)



            # sigmas = torch.zeros((nb_samples_lambda, n), dtype=torch.int)
            # fitnesses = torch.zeros(nb_samples_lambda)
            #
            # for i in range(nb_samples_lambda):
            #     # sample sigma_i from Plackett luce
            #     sigmas[i] = self.pl.sample_permutation(torch.exp(w_log))
            #     fitnesses[i] = fitness_function(sigmas[i])
            #
            #     if fitnesses[i] < best_fitness:
            #         best_fitness = fitnesses[i]
            #         sigma_best = sigmas[i]
            #
            # # w_log = ...
            #
            # avg_fitness = torch.mean(fitnesses)
            # self.utility.print_score(ctr, best_fitness, avg_fitness, nb_samples_lambda)
            # self.utility.print_array((w_log), ctr, frequency=10)

            ctr += 1
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
