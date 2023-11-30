# data comes from: http://comopt.ifi.uni-heidelberg.de/software/LOLIB/
# The link was discussed in paper:
# "A benchmark library and a comparison of heuristic  methods for the linear ordering problem"

# from scipy.io import loadmat
# x = loadmat('be75eec.mat')
# print(x)


import r0123456
import numpy as np

from benchmark_lolib import Benchmark
from placket_luce import PlackettLuce


def run_experiment():
    print("*******************************************************************")
    print("Running experiment with parameters:")

    # lr, nb_samples_lambda, numIters, U
    lr = 0.1
    nb_samples_lambda = 100
    numIters = 1000
    U = PlackettLuce.U_identity


    a = r0123456.r0123456(lr, nb_samples_lambda, numIters, U)
    benchmark = Benchmark(filename, normalize=True)
    best_fitness = a.optimize(benchmark)

    return best_fitness


if __name__ == "__main__":
    seed = 123456
    np.random.seed(seed)
    # filename = "./tour50.csv"
    filename = "./benchmarks/be75eec.mat"

    # Set parameters
    run_experiment()
