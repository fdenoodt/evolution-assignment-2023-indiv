# data comes from: http://comopt.ifi.uni-heidelberg.de/software/LOLIB/
# The link was discussed in paper:
# "A benchmark library and a comparison of heuristic  methods for the linear ordering problem"


# 3.2 Individual phase (40h)
# In the individual phase you continue from the code that you prepared in the group phase with your group.
# From this point onwards, the project is an individual assignment. Cooperation is forbidden in this phase.
# The goal of this phase is that you develop an advanced, optimized evolutionary algorithm for the traveling salesperson problem. You could implement additional or more advanced initialization heuristics, selection schemes, elimination mechanisms, variation operators, local search operators, population enrichment
# schemes, parameter self-adaptivity, optimized hyperparameter search, diversity-promoting schemes, and so
# on. You should also write an individual final report using the final report template.
# The Python code and final report should be turned in via Toledo by December 31, 2023 at 18:00 CET.


import r0123456
import numpy as np

# from benchmark_lolib import Benchmark
from benchmark_tsp import Benchmark
from placket_luce import PlackettLuce, VanillaPdf, PdfRepresentation, ConditionalPdf
from plackett_luce_algorithm import PlackettLuceAlgorithm


def run_experiment():
    print("*******************************************************************")
    print("Running experiment with parameters:")

    # lr, nb_samples_lambda, numIters, U
    lr = 0.9
    nb_samples_lambda = 100
    numIters = 1_000_000
    U = PlackettLuce.U_identity

    benchmark = Benchmark(filename, normalize=True, maximise=False)
    pdf: PdfRepresentation = VanillaPdf(benchmark.permutation_size())
    # pdf: PdfRepresentation = ConditionalPdf(benchmark.permutation_size())

    algorithm = PlackettLuceAlgorithm(lr, nb_samples_lambda, U, benchmark, pdf)
    a = r0123456.r0123456(algorithm, numIters)
    best_fitness = a.optimize()

    return best_fitness


if __name__ == "__main__":
    seed = 123456
    np.random.seed(seed)
    filename = "./tour50.csv"
    # filename = "./benchmarks/be75eec.mat"

    # Set parameters
    run_experiment()
