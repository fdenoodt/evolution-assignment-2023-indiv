import numpy as np


class PlackettLuce:

    def __init__(self, num_cities):
        self.num_cities = num_cities


    def U_identity(self, xs):
        return xs

    def U_normalize(self, xs):
        return xs / np.sum(xs)

    def U_super_linear(self, xs):  # xs are fitnesses
        # Sort the samples from the best to the worst in
        # terms of fitness and set 𝜇 = 𝜆/2. Assign null utility to the 𝜇 worst
        # samples, while, for the remaining ones, temporarily assign to the
        # 𝑖–th best sample exp(𝑖) points of utility and, finally, normalize the
        # utilities of the best 𝜇 samples. This utility function makes PL-GS
        # invariant for monotonic transformations of the objective function
        # and it is inspired by weights used in the CMA-ES algorithm [16].

        # this is wrong, it returns the sorted xs, not the sorted indices
        # mu = len(xs) / 2
        # xs = np.sort(xs)  # np.sort is ascending, so best fitness is last
        # xs[:int(mu)] = 0  # set worst mu to 0
        # xs[int(mu):] = np.exp(xs[int(mu):])  # set best mu to exp(i)
        # xs = xs / np.sum(xs)  # normalize

        # this is correct
        mu = len(xs) / 2
        sorted_indices = np.argsort(xs)
        adjusted_xs = np.zeros_like(xs)
        adjusted_xs[sorted_indices[:int(mu)]] = 0
        adjusted_xs[sorted_indices[int(mu):]] = np.exp(xs[sorted_indices[int(mu):]])
        # adjusted_xs = adjusted_xs / np.sum(adjusted_xs)

        return adjusted_xs

    def sample_permutation(self, w):
        n = self.num_cities
        sigma = np.zeros(n, dtype=int)
        used_nodes = np.zeros(n, dtype=bool)

        for i in range(n):
            node = self.sample_node(w, used_nodes)  # should return one city
            sigma[i] = node
            used_nodes[node] = True

        return sigma

    def sample_node(self, w, used_nodes):
        # compute probabilities: its the values in w, except its zero for used nodes
        probabilities = w.copy()
        probabilities[used_nodes] = 0
        probabilities /= np.sum(
            probabilities)  # TODO: COULD BE THAT W'S ARE STILL LOGS SO NEED TO EXP THEM OR VICE VERSA

        # check if probabilities contain NaNs
        if np.isnan(probabilities).any():
            assert False

        # sample from probabilities
        node = np.random.choice(self.num_cities, p=probabilities)

        return node

    def calc_w_log_p_partial(self, w_log, sigma, i):
        n = len(sigma)

        intermediate_result = 0  # calc \Sigma (1/sum)
        for k in range(i):
            sum = 0  # calc sum in denominator
            for j in range(k, n):
                sum += np.exp(w_log[sigma[j]])

            intermediate_result += 1 / sum

        return 1 - np.exp(w_log[sigma[i]]) * intermediate_result

    def calc_w_log_p(self, w_log, sigma):
        # Calculates all partial derivatives for a sample sigma
        n = len(sigma)
        gradient = np.zeros_like(w_log)

        for i in range(n):
            gradient[sigma[i]] = self.calc_w_log_p_partial(w_log, sigma, i)

        return gradient

    def calc_w_log_F(self, w_log, fitnesses, delta_w_log_ps, U, nb_samples_lambda, ):
        gradient = np.zeros_like(w_log)

        f_vals = U(fitnesses)  # list of scalar with len nb_samples_lambda

        # old way, slow
        # for i in range(nb_samples_lambda):
        #     gradient += f_val * delta_w_log_ps[i]  # scalar * vector

        gradient = np.dot(f_vals, delta_w_log_ps)  # f_vals is a vector, delta_w_log_ps is a matrix
        # f_vals[i] will multiply the i'th row of delta_w_log_ps, then sum over all rows, somehow it works

        gradient /= nb_samples_lambda

        return gradient
