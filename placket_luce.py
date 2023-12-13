from abc import ABC, abstractmethod

import numpy as np


class PdfRepresentation(ABC):
    def __init__(self, n):
        self.n = n

    @abstractmethod
    def sample_permutations(self, w, nb_samples_lambda):
        pass

    @abstractmethod
    def calc_gradients(self, w_log, sigmas):
        pass


class VanillaPdf(PdfRepresentation):
    def __init__(self, n):
        super().__init__(n)

    def sample_permutations(self, w, nb_samples_lambda):
        n = len(w)
        logits = w

        u = np.random.rand(nb_samples_lambda, n)  # shape: (nb_samples_lambda, n)
        g = logits - np.log(-np.log(u))  # shape: (nb_samples_lambda, n)

        res = np.argsort(-g, axis=1)  # shape: (nb_samples_lambda, n)
        return res

    @staticmethod
    def calc_w_log_p_partial(w_log, sigma, i):
        if i > 0:
            sums = [np.sum(np.exp(w_log[sigma[k:]])) for k in range(i)]
            intermediate_result = np.sum([1 / sum for sum in sums])
            return 1 - np.exp(w_log[sigma[i]]) * intermediate_result
        else:
            return 1

    @staticmethod
    def inner_loop(i, sigmas, w_log, n):
        js = np.arange(n)
        gradient = np.zeros_like(w_log)
        gradient[sigmas[i][js]] = np.array(
            [VanillaPdf.calc_w_log_p_partial(w_log, sigmas[i], j) for j in range(n)])
        return gradient

    @staticmethod
    def calc_w_log_ps(w_log, sigmas):
        # Calculates all partial derivatives for a list of samples sigmas
        n = len(sigmas[0])
        nb_samples_lambda = len(sigmas)

        gradients = np.zeros((nb_samples_lambda, n))
        iss = np.arange(nb_samples_lambda)
        gradients[iss] = np.array(
            [VanillaPdf.inner_loop(i, sigmas, w_log, n) for i in range(nb_samples_lambda)])

        return gradients  # shape: (nb_samples_lambda, n)

    def calc_gradients(self, w_log, sigmas):
        """
        Calculates the gradient of the log probability of the Plackett-Luce model
        :param w_log: log of the weights (length n)
        :param sigmas: list of sampled permutations (length nb_samples_lambda)
        :return: gradient of the log probability of the Plackett-Luce model. Shape: (nb_samples_lambda, n)
        """
        return VanillaPdf.calc_w_log_ps(w_log, sigmas)


class PlackettLuce:
    def __init__(self, U, benchmark):
        assert benchmark.maximise is True  # false not implemented yet
        assert callable(U)

        self.U = U
        self.benchmark = benchmark

    @staticmethod
    def U_identity(xs):
        return xs

    @staticmethod
    def U_normalize(xs):
        return xs / np.sum(xs)

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
        # sorted_indices = np.argsort(xs)
        # adjusted_xs = np.zeros_like(xs)
        # adjusted_xs[sorted_indices[:int(mu)]] = 0
        # adjusted_xs[sorted_indices[int(mu):]] = np.exp(xs[sorted_indices[int(mu):]])
        # adjusted_xs = adjusted_xs / np.sum(adjusted_xs)

        # for maximisation:
        mu = len(xs) / 2
        sorted_indices = np.argsort(xs)
        adjusted_xs = np.zeros_like(xs)
        adjusted_xs[sorted_indices[:int(mu)]] = np.exp(xs[sorted_indices[:int(mu)]])
        adjusted_xs[sorted_indices[int(mu):]] = 0  # now final ones are the worst
        adjusted_xs = adjusted_xs / np.sum(adjusted_xs)

        return adjusted_xs

    @staticmethod
    def calc_w_log_F(U, w_log, fitnesses, delta_w_log_ps, nb_samples_lambda, ):
        # gradient = np.zeros_like(w_log)

        f_vals = U(fitnesses)  # list of scalar with len nb_samples_lambda
        assert len(f_vals) == nb_samples_lambda
        assert len(delta_w_log_ps) == nb_samples_lambda

        gradient = np.dot(f_vals, delta_w_log_ps)  # f_vals is a vector, delta_w_log_ps is a matrix
        gradient /= nb_samples_lambda  # ??? maybe this is wrong? i think they dont do it in the code

        return gradient


if __name__ == "__main__":
    # Example usage
    # n = 5
    # expw = np.random.rand(n)  # Replace this with your actual data
    # x = np.random.permutation(n)  # Example ranking
    # g = np.zeros_like(expw, dtype=float)
    #
    # success = PlackettLuce.grad_log_prob(g, x, expw)
    #
    # grad2 = PlackettLuce.calc_w_log_ps(np.log(expw), np.array([x]))
    #
    # assert np.allclose(g, grad2[0])
    #
    # if success:
    #     print("Gradient computation successful.")
    #     print("Gradient values:", g)
    # else:
    #     print("Error encountered during gradient computation.")

    # # test calc_w_log_ps
    # w_log = np.array([1, 2, 3])
    # sigmas = np.array([[0, 1, 2], [1, 0, 2]])
    # gradients = PlackettLuce.calc_w_log_ps(w_log, sigmas)

    # test calc_gradients in PlackettLuce
    n = 5
    expw = np.random.rand(n)  # Replace this with your actual data
    x = np.random.permutation(n)  # Example ranking
    g = np.zeros_like(expw, dtype=float)

    pdf = VanillaPdf(n)

    gradients = pdf.calc_gradients(np.log(expw), np.array([x]))
    print(gradients)
