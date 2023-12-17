from abc import ABC, abstractmethod

import numpy as np


class PdfRepresentation(ABC):
    def __init__(self, n, w_log):
        self.n = n
        self.w_log = w_log

    @abstractmethod
    def sample_permutations(self, nb_samples_lambda):
        pass

    @abstractmethod
    def calc_gradients(self, sigmas):
        pass

    @abstractmethod
    def update_w_log(self, delta_w_log_F, lr):
        pass


class VanillaPdf(PdfRepresentation):
    def __init__(self, n, w_log=None):
        if w_log is None:
            w_log = np.zeros(n)  # w is w_tilde
            print("*" * 80)
            print("w_log is None, initializing with zeros!")
            print("*" * 80)
        else:
            assert len(w_log) == n

        super().__init__(n, w_log)

    def sample_permutations(self, nb_samples_lambda):
        logits = self.w_log  # shape: (n,)
        # TODO: w_log is correct, but np.exp(w_log) results in much faster convergence
        n = len(logits)

        u = np.random.rand(nb_samples_lambda, n)  # shape: (nb_samples_lambda, n)
        g = logits - np.log(-np.log(u))  # shape: (nb_samples_lambda, n)

        res = np.argsort(-g, axis=1)  # shape: (nb_samples_lambda, n)
        return res

    def sample_permutation(self):
        w = np.exp(self.w_log)
        n = len(w)
        probabilities = w / np.sum(w)
        permutation = np.random.choice(n, size=n, replace=False, p=probabilities)
        return permutation

    def sample_permutations_slow(self, nb_samples_lambda):
        permutations = np.array([self.sample_permutation() for _ in range(nb_samples_lambda)])
        return permutations

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

    def calc_gradients(self, sigmas):
        """
        Calculates the gradient of the log probability of the Plackett-Luce model
        :param w_log: log of the weights (length n)
        :param sigmas: list of sampled permutations (length nb_samples_lambda)
        :return: gradient of the log probability of the Plackett-Luce model. Shape: (nb_samples_lambda, n)
        """
        return VanillaPdf.calc_w_log_ps(self.w_log, sigmas)

    def update_w_log(self, delta_w_log_F, lr):
        self.w_log = self.w_log + (lr * delta_w_log_F)  # "+" for maximization, "-" for minimization


class ConditionalPdf(PdfRepresentation):
    def __init__(self, n, w_log=None):
        # W[i, j] is P(i | j)
        if w_log is None:
            w_log = np.zeros((n, n))
            print("*" * 80)
            print("w_log is None, initializing with zeros!")
            print("*" * 80)
        else:
            assert w_log.shape == (n, n)
        super().__init__(n, w_log)

    def sample_node_given(self, j, permutation_so_far, L):
        """
        Samples a permutation given the previous node
        :param j: the previously sampled node
        :param permutation_so_far: the permutation so far, i.e. the first L-1 nodes.
        Those nodes are fixed and will not be sampled again.
        """

        # w_log_giv_j = self.w_log[:, j]
        # w_giv_j = np.exp(w_log_giv_j)
        # n = len(w_giv_j)
        # probabilities = w_giv_j / np.sum(w_giv_j)
        # permutation = np.random.choice(n, size=n, replace=False, p=probabilities)
        # return permutation

        # return only single node
        w_log_giv_j = self.w_log[:, j]
        w_giv_j = np.exp(w_log_giv_j)

        # remove nodes that are already in the permutation
        w_giv_j[permutation_so_far] = 0

        probabilities = w_giv_j / np.sum(w_giv_j)
        # sample one node from the marginal distribution
        node = np.random.choice(self.n, size=1, replace=False, p=probabilities)[0]
        return node

    def sample_permutation_marginal(self):
        # need to calc For all i: P(i) = sum_j P(i | j) // bayes factorization
        p_is = np.zeros(self.n)
        for i in range(self.n):
            p_is[i] = np.sum(np.exp(self.w_log[i, :]))

        probabilities = p_is / np.sum(p_is)

        # sample one node from the marginal distribution
        permutation = np.random.choice(self.n, size=1, replace=False, p=probabilities)[0]
        return permutation

    def sample_permutation(self):
        permutation = np.zeros(self.n, dtype=int)
        permutation[0] = self.sample_permutation_marginal()
        for i in range(1, self.n):
            permutation[i] = self.sample_node_given(permutation[i - 1], permutation[:i], i)
        return permutation

        # TODO:
        # permutation[1:] = self.sample_permutation_given(permutation[:-1])

    def sample_permutations(self, nb_samples_lambda):
        permutations = np.array([self.sample_permutation() for _ in range(nb_samples_lambda)])
        return permutations

    def calc_gradients(self, sigmas):
        n = len(sigmas[0])
        nb_samples_lambda = len(sigmas)

        gradients = np.zeros((nb_samples_lambda, n))
        return gradients  # shape (nb_samples_lambda, n)

    def update_w_log(self, delta_w_log_F, lr):
        # self.w_log = self.w_log + (lr * delta_w_log_F)  # "+" for maximization, "-" for minimization
        # not implemented yet exception
        raise NotImplementedError


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
    def calc_w_log_F(U, fitnesses, delta_w_log_ps, nb_samples_lambda, ):
        f_vals = U(fitnesses)  # list of scalar with len nb_samples_lambda
        assert len(f_vals) == nb_samples_lambda
        assert len(delta_w_log_ps) == nb_samples_lambda

        gradient = np.dot(f_vals, delta_w_log_ps)  # f_vals is a vector, delta_w_log_ps is a matrix
        gradient /= nb_samples_lambda  # ??? maybe this is wrong? i think they dont do it in the code

        return gradient


if __name__ == "__main__":
    # test sample_permutations
    n = 5

    # Example vanilla pdf
    # w = np.random.rand(n)
    w = np.array([1, 110, 220, 455, 999])
    pdf = VanillaPdf(n, np.log(w))
    sigmas = pdf.sample_permutations(10)
    sigmas2 = pdf.sample_permutations_slow(10)
    print(sigmas)
    print(sigmas2)

    # Example conditional pdf
    pdf = ConditionalPdf(n)
    sigmas = pdf.sample_permutations(10)
    print(sigmas)

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

    # # test calc_gradients in PlackettLuce
    # n = 5
    # expw = np.random.rand(n)  # Replace this with your actual data
    # x = np.random.permutation(n)  # Example ranking
    # g = np.zeros_like(expw, dtype=float)
    #
    # pdf = VanillaPdf(n)
    #
    # gradients = pdf.calc_gradients(np.log(expw), np.array([x]))
    # print(gradients)
