import reporter as Reporter
from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from placket_luce import PlackettLuce
from utility import Utility

import numpy as np


class r0698535:
    def __init__(self, algorithm: AbstractAlgorithm, numIters):
        self.reporter_name = self.__class__.__name__
        self.algorithm = algorithm
        self.numIters = numIters

    def optimize(self):
        return self.algorithm.optimize(self.numIters, self.reporter_name)

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
