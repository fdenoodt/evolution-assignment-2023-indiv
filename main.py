# data comes from: http://comopt.ifi.uni-heidelberg.de/software/LOLIB/
# The link was discussed in paper:
# "A benchmark library and a comparison of heuristic  methods for the linear ordering problem"

# from scipy.io import loadmat
# x = loadmat('be75eec.mat')
# print(x)


import r0123456
import numpy as np


def run_experiment():
    print("*******************************************************************")
    print("Running experiment with parameters:")

    a: r0123456 = r0123456.r0123456()

    best_fitness = a.optimize(filename)
    return best_fitness


# The main function can be used to test your code.
if __name__ == "__main__":
    seed = 123456
    np.random.seed(seed)
    filename = "./tour50.csv"

    # Set parameters
    run_experiment()
