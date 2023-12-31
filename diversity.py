import numpy as np

from ScoreTracker import ScoreTracker
from benchmark_tsp import Benchmark
from local_search import LocalSearch
from selection import Selection
from utility import Utility
from variation import Variation


#
#
# if __name__ == "__main__":
#     print("*" * 20)
#     print("Test migration")
#     n = 8
#     popul_size = 4
#     islands = [Island(i, None, popul_size, n) for i in range(3)]
#     print("Before migration")
#     print(islands[0].population)
#     print(islands[1].population)
#     print(islands[2].population)
#     Island.migrate(islands, popul_size, percentage=0.5)
#     print("After migration")
#     print(islands[0].population)
#     print(islands[1].population)
#     print(islands[2].population)
#
#     print("*" * 20)
#     print("Test run_epochs")
#     benchmark = Benchmark("./tour750.csv", normalize=True, maximise=False)
#     n = benchmark.permutation_size()
#     popul_size = 100
#     islands = [Island(i, lambda x: np.random.rand(len(x)), popul_size, n) for i in range(5)]
#     print("Before run_epochs")
#
#     import time
#
#     time1 = time.time()
#
#     Island.run_epochs(5, islands,
#                       selection=lambda population, fitnesses: population,
#                       elimination=lambda population, fitnesses: population,
#                       mutation=lambda offspring: offspring,
#                       crossover=lambda selected: selected,
#                       score_tracker=ScoreTracker(n, maximize=False, keep_running_until_timeup=False, numIters=1,
#                                                  reporter_name="test",
#                                                  benchmark=benchmark),
#                       ctr=0)
#
#     time2 = time.time()
#
#     print("Time to run_epochs:", time2 - time1)

# TODO: think about why the best_fitness is signficantly lower here than when running main.py
