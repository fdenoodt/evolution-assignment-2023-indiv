import reporter as Reporter
from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from algorithm_wrapper import AlgorithmWrapper, HyperparamsPlackettLuceAlgorithm, HyperparamsEvolAlgorithm
from placket_luce import PlackettLuce, PdfRepresentation, ConditionalPdf, VanillaPdf
from utility import Utility

import numpy as np


class r0698535:
    def __init__(self):
        self.reporter_name = self.__class__.__name__
        self.run_plackett_luce = False

    def optimize(self, filename):
        if self.run_plackett_luce:
            assert filename.endswith("tour50.csv"), "PL-GS only supports tour50.csv"

            benchmark_filename = filename
            pdf: PdfRepresentation = VanillaPdf(n=50)
            # pdf: PdfRepresentation = ConditionalPdf(n=50)
            hyperparams = HyperparamsPlackettLuceAlgorithm(pdf)
            return AlgorithmWrapper.run_experiment_plackett_luce(hyperparams, benchmark_filename, pdf,
                                                                 self.reporter_name)

        else:  # run evol algorithm
            if filename.endswith("tour50.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                # *****./tour50.csv********* BEST PARAMS *****
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1  # 3
                hyperparams.k = 3
                hyperparams.mutation_rate = 0.1
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("2-opt", 1)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=20)  # only runs for 20 seconds, no need for longer

            elif filename.endswith("tour100.csv") or filename.endswith("tour200.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 3  # strange ...
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("2-opt", 1)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)  # will run for 5 minutes (default)
            elif filename.endswith("tour500.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 10
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)

            elif filename.endswith("tour750.csv"):
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 25  # strange ...
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 25
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)  # will run for 5 minutes (default)

            else:  # for 1000 tours or any other file
                print("*" * 100)
                print("RUNNING OPTIMAL CONFIG FOR 1_000 TOURS!!!!")
                benchmark_filename = filename
                hyperparams = HyperparamsEvolAlgorithm()
                hyperparams.popul_size = 50
                hyperparams.offspring_size_multiplier = 1
                hyperparams.k = 25  # strange ... selection pressure is too high (popul is 50)
                hyperparams.mutation_rate = 0.2
                hyperparams.migrate_after_epochs = 5
                hyperparams.migration_percentage = 0.05
                hyperparams.merge_after_percent_time_left = 0.5
                hyperparams.fitness_sharing_subset_percentage = 0.05
                hyperparams.alpha = 1
                hyperparams.local_search = ("insert_random_node", 0.5)

                return AlgorithmWrapper.repeat_experiment(hyperparams, benchmark_filename,
                                                          reporter_name=self.reporter_name, nb_repeats=1,
                                                          max_duration=None)


if __name__ == "__main__":
    r = r0698535()
    # print(r.optimize("./tour50.csv"))
    # print(r.optimize("./tour750.csv"))
    print(r.optimize("./tour1000.csv"))
    print(r.optimize("./tour500.csv"))
