from abc import ABC, abstractmethod
import numpy as np


# abstract base class for Benchmark

class AbstractAlgorithm(ABC):

    @abstractmethod
    # variable number of arguments
    def optimize(self, *args):
        pass
