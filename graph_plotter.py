import matplotlib.pyplot as plt
# import tikzplotlib
import os
import numpy as np


if __name__ == "__main__":
    GraphPlotter.make_bar_graph("./BARS/50_tours/", 500)

    # GraphPlotter.read_file_and_make_graph("r0123456.csv")

    # # best params:     run_experiment(100, 1, 10, 0.05, filename, save_to)
    # popul_size = 100
    # offspring_size_multiplier = 1
    # k = 10
    # mutation_rate = 0.05
    #
    # results = GraphPlotter.read_report_file("tour50_FINAL_.csv")
    # results0 = GraphPlotter.read_report_file("tour50_FINAL_0.csv")
    # results1 = GraphPlotter.read_report_file("tour50_FINAL_1.csv")
    # results2 = GraphPlotter.read_report_file("tour50_FINAL_2.csv")
    # results3 = GraphPlotter.read_report_file("tour50_FINAL_3.csv")
    # results4 = GraphPlotter.read_report_file("tour50_FINAL_4.csv")
    #
    # numIterationss, timeElapseds, meanObjectives, bestObjectives = results
    # numIterationss0, timeElapseds0, meanObjectives0, bestObjectives0 = results0
    # numIterationss1, timeElapseds1, meanObjectives1, bestObjectives1 = results1
    # numIterationss2, timeElapseds2, meanObjectives2, bestObjectives2 = results2
    # numIterationss3, timeElapseds3, meanObjectives3, bestObjectives3 = results3
    # numIterationss4, timeElapseds4, meanObjectives4, bestObjectives4 = results4
    #
    # data_iterations = [numIterationss0, numIterationss1, numIterationss2, numIterationss3, numIterationss4]
    # data_time = [timeElapseds0, timeElapseds1, timeElapseds2, timeElapseds3, timeElapseds4]
    # data_mean = [meanObjectives0, meanObjectives1, meanObjectives2, meanObjectives3, meanObjectives4]
    # data_best = [bestObjectives0, bestObjectives1, bestObjectives2, bestObjectives3, bestObjectives4]
    #
    # # create_line_graph(data_iterations[0], data_mean[0], data_best[0], data_time[0],
    # #                   "Number of seconds",
    # #                   "Objective value",
    # #                   "Mean and best objective value over time", "mean_objective_value")
    # #
    # # # only first 10% of the iterations
    # # create_line_graph(numIterationss[:int(len(numIterationss) * 0.1)],
    # #                   meanObjectives[:int(len(meanObjectives) * 0.1)],
    # #                   bestObjectives[:int(len(bestObjectives) * 0.1)],
    # #                   timeElapseds[:int(len(timeElapseds) * 0.1)],
    # #                   "Number of seconds",
    # #                   "Objective value",
    # #                   "Mean and best objective value over time (first 10% of iterations)",
    # #                   "mean_objective_value_first_10_percent")
    #
    # GraphPlotter.compare_best(numIterationss3, data_best, timeElapseds3,
    #                           "Number of seconds",
    #                           "Objective value",
    #                           "Best objective value over time - 6 runs",
    #                           "variance")
    #
    # GraphPlotter.compare_best(numIterationss3[:int(len(numIterationss3) * 0.1)], data_best,
    #                           timeElapseds3[:int(len(timeElapseds3) * 0.1)],
    #                           "Number of seconds",
    #                           "Objective value",
    #                           "Best objective value over time (first 10% of iterations) - 6 runs",
    #                           "variance_10_percent")
