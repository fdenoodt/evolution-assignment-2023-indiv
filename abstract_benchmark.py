from abc import ABC, abstractmethod
import numpy as np


# abstract base class for Benchmark

class AbstractBenchmark(ABC):
    def __init__(self, matrix, normalize):
        self.normalizing_constant = 1
        self.matrix = matrix
        if normalize:
            self.matrix, self.normalizing_constant = AbstractBenchmark.normalize_matrix(self.matrix)

    def permutation_size(self):
        return self.matrix.shape[0]

    @staticmethod
    def normalize_matrix(matrix):
        # normalize distance matrix to be between 0 and 1
        # it makes the w's smaller and thus less likely to overflow
        constant = np.max(matrix)
        distanceMatrix = matrix / constant
        return distanceMatrix, constant

    @abstractmethod
    def compute_fitness(self, population):
        pass
