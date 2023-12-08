import torch
# import numpy as np

class PlackettLuce:

    def __init__(self, U, maximise=True):
        self.U = U

    @staticmethod
    def U_identity(xs):
        return xs

    @staticmethod
    def U_normalize(xs):
        return xs / torch.sum(xs)

    @staticmethod
    def U_super_linear(xs):  # xs are fitnesses
        # Sort the samples from the best to the worst in
        # terms of fitness and set 𝜇 = 𝜆/2. Assign null utility to the 𝜇 worst
        # samples, while, for the remaining ones, temporarily assign to the
        # 𝑖–th best sample exp(𝑖) points of utility and, finally, normalize the
        # utilities of the best 𝜇 samples. This utility function makes PL-GS
        # invariant for monotonic transformations of the objective function
        # and it is inspired by weights used in the CMA-ES algorithm [16].

        # in case of minimization:
        # mu = len(xs) / 2
        # sorted_indices = torch.argsort(xs)
        # adjusted_xs = torch.zeros_like(xs)
        # adjusted_xs[sorted_indices[:int(mu)]] = 0
        # adjusted_xs[sorted_indices[int(mu):]] = torch.exp(xs[sorted_indices[int(mu):]])
        # adjusted_xs = adjusted_xs / torch.sum(adjusted_xs)

        # for maximisation:
        mu = len(xs) / 2
        sorted_indices = torch.argsort(xs)
        adjusted_xs = torch.zeros_like(xs)
        adjusted_xs[sorted_indices[:int(mu)]] = torch.exp(xs[sorted_indices[:int(mu)]])
        adjusted_xs[sorted_indices[int(mu):]] = 0  # now final ones are the worst
        adjusted_xs = adjusted_xs / torch.sum(adjusted_xs)

        return adjusted_xs

    @staticmethod
    def sample_permutation(w):
        n = len(w)
        logits = w  # TODO: maybe expects w_log instead of w

        u = torch.rand(n)
        g = logits - torch.log(-torch.log(u))

        # causes numba error:
        # res = np.argsort(-g, kind='stable') # negativized because descending sorting is required
        res = torch.argsort(-g)

        return res

    # def sample_permutation(self, w):
    #     n = len(w)
    #     sigma = torch.zeros(n, dtype=torch.int)
    #     used_nodes = torch.zeros(n, dtype=torch.bool)
    #
    #     for i in range(n):
    #         node = self.sample_node(w, used_nodes)  # should return one city
    #         sigma[i] = node
    #         used_nodes[node] = True
    #
    #     return sigma
    #
    # def sample_node(self, w, used_nodes):
    #     # compute probabilities: its the values in w, except its zero for used nodes
    #     probabilities = w.clone()
    #     probabilities[used_nodes] = 0
    #     probabilities /= torch.sum(
    #         probabilities)
    #
    #     # check if probabilities contain NaNs
    #     if torch.isnan(probabilities).any():
    #         assert False
    #
    #     # sample from probabilities
    #     n = len(probabilities)
    #     node = np.random.choice(n, p=probabilities.numpy())
    #
    #     return node

    @staticmethod
    def calc_w_log_p_partial(w_log, sigma, i):
        n = len(sigma)

        intermediate_result = 0  # calc \Sigma (1/sum)
        for k in range(i):
            sum = 0  # calc sum in denominator
            for j in range(k, n):
                sum += torch.exp(w_log[sigma[j]])

            intermediate_result += 1 / sum

        return 1 - torch.exp(w_log[sigma[i]]) * intermediate_result

    @staticmethod
    def calc_w_log_p(w_log, sigma):
        # Calculates all partial derivatives for a sample sigma
        n = len(sigma)
        gradient = torch.zeros_like(w_log)

        for i in range(n):
            gradient[sigma[i]] = PlackettLuce.calc_w_log_p_partial(w_log, sigma, i)

        return gradient

    def calc_w_log_F(self, w_log, fitnesses, delta_w_log_ps, nb_samples_lambda, ):
        gradient = torch.zeros_like(w_log)

        f_vals = self.U(fitnesses)  # list of scalar with len nb_samples_lambda
        assert len(f_vals) == nb_samples_lambda
        assert len(delta_w_log_ps) == nb_samples_lambda

        # old way, slow
        for i in range(nb_samples_lambda):
            gradient += f_vals[i] * delta_w_log_ps[i]  # scalar * vector

        # gradient = torch.dot(f_vals, delta_w_log_ps)  # f_vals is a vector, delta_w_log_ps is a matrix
        # f_vals[i] will multiply the i'th row of delta_w_log_ps, then sum over all rows, somehow it works

        gradient /= nb_samples_lambda

        return gradient
