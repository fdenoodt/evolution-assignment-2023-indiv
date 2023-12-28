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


import r0698535
import numpy as np

# from benchmark_lolib import Benchmark
from benchmark_tsp import Benchmark
from evol_algorithm import EvolAlgorithm
from placket_luce import PlackettLuce, VanillaPdf, PdfRepresentation, ConditionalPdf
from plackett_luce_algorithm import PlackettLuceAlgorithm
from graph_plotter import GraphPlotter


def run_experiment(hyperparams, benchmark_filename):
    print("*******************************************************************")
    print("Running experiment with parameters:")
    print(hyperparams.__dict__)

    # csv_filename is based on hyperparams and benchmark_filename
    GraphPlotter.mkdir(f"./{benchmark_filename[:-4]}")
    csv_filename = (f"./{benchmark_filename[:-4]}/popul_size={hyperparams.popul_size},"
                    f"offsp_sz_multipl={hyperparams.offspring_size_multiplier},k={hyperparams.k},"
                    f"mut_r={hyperparams.mutation_rate},nb_isl={hyperparams.nb_islands},"
                    f"migr_aftr_ep={hyperparams.migrate_after_epochs},migr_perc={hyperparams.migration_percentage},"
                    f"mrge_aftr_perc_time_left={hyperparams.merge_after_percent_time_left},"
                    f"fit_shr_sbst_perc={hyperparams.fitness_sharing_subset_percentage},alph={hyperparams.alpha},"
                    f"local_search={hyperparams.local_search}")

    numIters = np.inf
    benchmark = Benchmark(benchmark_filename, normalize=False, maximise=False)

    algorithm = EvolAlgorithm(benchmark, hyperparams, csv_filename)
    a = r0698535.r0698535(algorithm, numIters)

    try:
        best_fitness = a.optimize()
        # best_fitness = 0
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        best_fitness = 0
    finally:
        # plot
        GraphPlotter.read_file_and_make_graph(f"{csv_filename}.csv")

    return best_fitness


class HyperparamsEvolAlgorithm:
    def __init__(self,
                 popul_size=100,
                 offspring_size_multiplier=1,
                 k=3,
                 mutation_rate=0.2,
                 # Islands
                 migrate_after_epochs=25, migration_percentage=0.1, merge_after_percent_time_left=0.5,
                 fitness_sharing_subset_percentage=0.1,  # higher is more accurate, but slower
                 alpha=1,  # used in fitness sharing
                 local_search=(None, None),
                 keep_running_until_timeup=True):
        self.popul_size = popul_size
        self.offspring_size_multiplier = offspring_size_multiplier
        self.k = k
        self.mutation_rate = mutation_rate

        self.nb_islands = 3  # Always fixed. one island per mutation function
        self.migrate_after_epochs = migrate_after_epochs
        self.migration_percentage = migration_percentage
        self.merge_after_percent_time_left = merge_after_percent_time_left  # eg 0.75 will merge when 75% of time is left

        self.fitness_sharing_subset_percentage = fitness_sharing_subset_percentage
        self.alpha = alpha

        self.local_search = local_search  # (None, None), ("2-opt", 1), ("insert_random_node", 0.1) ...
        # 2nd param is param for local search:
        # eg nb_nodes_to_insert_percent=0.1 for local_search="insert_random_node", jump_size=1 for local_search="2-opt"

        self.keep_running_until_timeup = keep_running_until_timeup


class HyperparamsPlackettLuceAlgorithm:
    def __init__(self,
                 lr=0.01,
                 nb_samples_lambda=100,
                 U=PlackettLuce.U_identity,
                 keep_running_until_timeup=True):
        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda
        self.U = U
        self.keep_running_until_timeup = keep_running_until_timeup
        # pdf: PdfRepresentation = VanillaPdf(benchmark.permutation_size())
        # pdf: PdfRepresentation = ConditionalPdf(benchmark.permutation_size())
        # algorithm = PlackettLuceAlgorithm(lr, nb_samples_lambda, U, benchmark, pdf)


def find_optimal_param(param_name, param_values, hyperparams, benchmark_filename):
    # *** POPUL_SIZE ***
    best_fitness = np.inf
    best_param = None
    for param_value in param_values:
        # hyperparams.popul_size = popul_size # not via string
        # hyperparams['popul_size'] = popul_size
        # via string:
        # exec(f"hyperparams.popul_size = {popul_size}")
        try:
            exec(f"hyperparams.{param_name} = {param_value}")

            fitness = run_experiment(hyperparams, benchmark_filename)
            if fitness < best_fitness:
                best_fitness = fitness
                best_param = param_value
        except Exception as e:
            append_to_file("best_params.txt", f"Error with {param_name} = {param_value}")
            # print hyper_params
            append_to_file("best_params.txt", str(hyperparams.__dict__))
            append_to_file("best_params.txt", str(e))
            continue

    # set best param
    exec(f"hyperparams.{param_name} = {best_param}")

    return best_param, best_fitness


def clear_file(filename):
    with open(filename, "w") as f:
        f.write("")


def append_to_file(filename, text):
    with open(filename, "a") as f:
        f.write(text + "\n")


def find_optimal_param_for_tsp(benchmark_filename, fixed_popul_size=100):
    # Set parameters
    hyperparams = HyperparamsEvolAlgorithm()  # start with default params, and change one at a time

    test_params = {
        "popul_size": [10, 100, 200, 500, 1000] if fixed_popul_size is None else [fixed_popul_size],
        "offspring_size_multiplier": [1, 2, 3],
        "k": [3, 5, 25],
        "mutation_rate": [0.05, 0.2, 0.4],
        "migrate_after_epochs": [25, 50],
        "migration_percentage": [0.05, 0.1],
        "merge_after_percent_time_left": [0.5, 0.75, 0.9],
        "fitness_sharing_subset_percentage": [0.05, 0.2, 0.5],
        "alpha": [1, 2, 0.5],
        "local_search": [(None, None), ("2-opt", 1), ("2-opt", 5),
                         ("insert_random_node", 0.1), ("insert_random_node", 0.5), ("insert_random_node", 1)]
    }

    # filename
    append_to_file(f"best_params.txt", f"\n\n\n*********{benchmark_filename}*********")

    for param_name, param_values in test_params.items():
        best_param, all_time_best_fitness = find_optimal_param(param_name, param_values, hyperparams,
                                                               benchmark_filename)
        print()
        print()
        print()
        print("*" * 100)
        print(f"Best {param_name} is {best_param} with fitness {all_time_best_fitness}")
        print("*" * 100)
        print()
        print()
        print()
        append_to_file("best_params.txt", f"Best {param_name} is {best_param} with fitness {all_time_best_fitness}")


if __name__ == "__main__":
    # benchmark_filename = "./benchmarks/be75eec.mat"

    # Params are chosen based on impact on fitness, one at a time
    # hyperparams.popul_size = 100
    # hyperparams.offspring_size_multiplier = 1
    # hyperparams.k = 3
    # hyperparams.mutation_rate = 0.2
    # hyperparams.nb_islands = 3, always fixed!
    # hyperparams.migrate_after_epochs = 25
    # hyperparams.migration_percentage = 0.05
    # hyperparams.merge_after_percent_time_left = 0.5
    # hyperparams.fitness_sharing_subset_percentage = 0.05
    # hyperparams.alpha = 1
    # hyperparams.local_search = "2-opt" & hyperparams.local_search_param = 1 together

    seed = 123456
    np.random.seed(seed)

    clear_file("best_params.txt")

    for benchmark_filename in ["./tour50.csv", "./tour200.csv", "./tour500.csv", "./tour750.csv", "./tour1000.csv"]:
        find_optimal_param_for_tsp(benchmark_filename)

    # do same but fix popul_size=100
    for benchmark_filename in ["./tour50.csv", "./tour200.csv", "./tour500.csv", "./tour750.csv", "./tour1000.csv"]:
        find_optimal_param_for_tsp(benchmark_filename)

