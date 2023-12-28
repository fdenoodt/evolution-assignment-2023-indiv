import matplotlib.pyplot as plt
# import tikzplotlib
import os
import numpy as np


class GraphPlotter:
    # @staticmethod
    # def graph_path():
    #     graphs_path = "./graphs/"
    #     return graphs_path

    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def create_line_graph(x, y1, y2, xticks, x_label, y_label, title, filename=None):
        # xticks to integers
        xticks = [int(xtick) for xtick in xticks]

        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        # set max y value to 0.5
        # plt.ylim(top=100_000, bottom=0)

        # legend: best and mean
        plt.legend(["Mean objective value", "Best objective value"])

        # Set the x-axis ticks with a subset of xticks
        num_xticks_to_display = 10  # You can adjust this number
        step = len(xticks) // num_xticks_to_display
        # plt.xticks(x[::step], xticks[::step])

        if filename is not None:
            filename = filename  # GraphPlotter.graph_path() + filename
            plt.savefig(filename + ".pdf")
            plt.savefig(filename + ".png")
            # tikzplotlib.save(filename + ".tex")

        plt.show()

    @staticmethod
    def compare_best(x, ys, xticks, x_label, y_label, title, filename=None):
        # GraphPlotter.mkdir()

        # xticks to integers
        xticks = [int(xtick) for xtick in xticks]

        for y in ys:
            y = y[:len(x)]
            plt.plot(x, y)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        # set max y value to 0.5
        plt.ylim(top=100_000, bottom=0)

        # Set the x-axis ticks with a subset of xticks
        num_xticks_to_display = 10  # You can adjust this number
        step = len(xticks) // num_xticks_to_display
        plt.xticks(x[::step], xticks[::step])

        if filename is not None:
            filename = filename  # GraphPlotter.graph_path() + filename
            plt.savefig(filename + ".pdf")
            # tikzplotlib.save(filename + ".tex")
            plt.savefig(filename + ".png")

        plt.show()

    @staticmethod
    def read_report_file(filename):
        # GraphPlotter.mkdir()

        data = []

        numIterationss = []
        timeElapseds = []
        meanObjectives = []
        bestObjectives = []

        with open(filename, "r") as inFile:
            for line in inFile:
                parts = line.strip().split()  # Split the line into parts based on delimiter
                parts = parts[0]
                parts = parts.split(',')
                if len(parts) >= 5:  # Ensure that there are enough elements to extract
                    numIterations = int(parts[0])
                    timeElapsed = float(parts[1])
                    meanObjective = float(parts[2])
                    bestObjective = float(parts[3])

                    numIterationss.append(numIterations)
                    timeElapseds.append(timeElapsed)
                    meanObjectives.append(meanObjective)
                    bestObjectives.append(bestObjective)

        return numIterationss, timeElapseds, meanObjectives, bestObjectives

    @staticmethod
    def read_file_and_make_graph(filename="r0123456.csv", target_dir="./graphs/"):
        target_file = f"{target_dir}/full/{filename}"  # eg ./graphs/tour50/r0123456.csv
        target_file_skip_25 = f"{target_dir}/skip_first_25/{filename}"
        target_file_first_10_percent = f"{target_dir}/first_10_percent/{filename}"

        dir = os.path.dirname(target_file)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)

        results = GraphPlotter.read_report_file(filename)
        numIterationss, timeElapseds, meanObjectives, bestObjectives = results

        GraphPlotter.create_line_graph(
            numIterationss, meanObjectives, bestObjectives, timeElapseds,
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time",
            f"{target_file}_mean_objective_value")

        dir = os.path.dirname(target_file_skip_25)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)
        GraphPlotter.create_line_graph(
            numIterationss[25:], meanObjectives[25:], bestObjectives[25:], timeElapseds[25:],
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time",
            f"{target_file_skip_25}_mean_objective_value")

        dir = os.path.dirname(target_file_first_10_percent)  # eg ./graphs/tour50/
        GraphPlotter.mkdir(dir)
        GraphPlotter.create_line_graph(
            numIterationss[:int(len(numIterationss) * 0.1)],
            meanObjectives[:int(len(meanObjectives) * 0.1)],
            bestObjectives[:int(len(bestObjectives) * 0.1)],
            timeElapseds[:int(len(timeElapseds) * 0.1)],
            "Number of iterations",
            "Objective value",
            "Mean and best objective value over time (first 10% of iterations)",
            f"{target_file_first_10_percent}_mean_objective_value")


if __name__ == "__main__":
    GraphPlotter.read_file_and_make_graph("r0123456.csv")

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
