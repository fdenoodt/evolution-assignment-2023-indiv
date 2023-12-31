from ScoreTracker import ScoreTracker
from abstract_algorithm import AbstractAlgorithm
from placket_luce import PlackettLuce


class PlackettLuceAlgorithm(AbstractAlgorithm):
    def __init__(self, lr, nb_samples_lambda, U, benchmark, pdf, keep_running_until_timeup, filename=None):
        assert not (benchmark.normalizing_constant == 1), \
            "Normalizing for PlackettLuceAlgorithm is required to prevent overflow, so it should be enabled"

        self.lr = lr
        self.nb_samples_lambda = nb_samples_lambda
        self.keep_running_until_timeup = keep_running_until_timeup

        self.pl = PlackettLuce(U, benchmark)
        self.pdf = pdf

        self.filename = filename  # used for saving the results (other than ./r01235..)

        super().__init__()

    def optimize(self, numIters, reporter_name, max_duration=None, *args):
        print("************** STARTING")
        print(reporter_name)

        # *args can be used as follows:
        # for arg in args:
        #     print(arg)

        # or via indexing:
        # print(args[0])

        # or via unpacking:
        # a, b, c = args

        n = self.pl.benchmark.permutation_size()
        f = self.pl.benchmark.compute_fitness
        keep_running_until_timeup = self.keep_running_until_timeup

        pdf = self.pdf
        maximize = self.pl.benchmark.maximise
        keep_running_until_timeup = keep_running_until_timeup

        score_tracker = ScoreTracker(n, maximize, keep_running_until_timeup, numIters, reporter_name, self.pl.benchmark,
                                     second_filename=self.filename, max_duration=max_duration)

        self.optimize_plackett_luce(f, self.lr, self.nb_samples_lambda, n, pdf, maximize, score_tracker)

    def optimize_plackett_luce(self, fitness_func, lr, nb_samples_lambda, n, pdf, maximize, score_tracker):
        ctr = 0
        while True:
            # Sample sigma_i from Plackett luce
            sigmas = pdf.sample_permutations(nb_samples_lambda)
            fitnesses = fitness_func(sigmas)

            delta_w_log_ps = pdf.calc_gradients(sigmas)
            best_fitness, mean_fitness, sigma_best = score_tracker.update_scores(
                fitnesses, sigmas, ctr,
                fitnesses_shared=None, # only used in EvolAlgorithm
                pdf=pdf,
                print_w=True)
            delta_w_log_F = PlackettLuce.calc_w_log_F(self.pl.U, fitnesses, delta_w_log_ps, nb_samples_lambda)
            pdf.update_w_log(delta_w_log_F, lr, maximize)

            ctr += 1
            is_done, time_left = score_tracker.utility.is_done_and_report(ctr, mean_fitness, best_fitness, sigma_best, write_to_file=True)
            if is_done:
                break

        return score_tracker.all_time_best_fitness
