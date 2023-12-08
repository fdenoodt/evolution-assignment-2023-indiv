import numpy as np


# import numpy as np

class PlackettLuce:

    def __init__(self, U, maximise=True):
        self.U = U

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
    def sample_permutation(w):
        n = len(w)
        logits = w  # TODO: maybe expects w_log instead of w

        u = np.random.rand(n)
        g = logits - np.log(-np.log(u))

        # causes numba error:
        # res = np.argsort(-g, kind='stable') # negativized because descending sorting is required
        res = np.argsort(-g)

        return res

    @staticmethod
    def sample_permutations(w, nb_samples_lambda):
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
    def calc_w_log_ps(w_log, sigmas):
        # Calculates all partial derivatives for a list of samples sigmas
        n = len(sigmas[0])
        nb_samples_lambda = len(sigmas)
        gradient_old = np.zeros((nb_samples_lambda, n))
        gradient = np.zeros((nb_samples_lambda, n))

        # old way, slow
        for i in range(nb_samples_lambda):
            for j in range(n):
                gradient_old[i][sigmas[i][j]] = PlackettLuce.calc_w_log_p_partial(w_log, sigmas[i], j)

        return gradient_old  # shape: (nb_samples_lambda, n)

    import numpy as np

    @staticmethod
    def grad_log_prob(g, x, expw):  # g is the gradient, x is the permutation, expw is the exponentiated w
        n = len(x)  # 50
        sIn = np.sum(expw)
        sOut = 1.0 / sIn
        g[x[0]] = 1.0 - expw[x[0]] * sOut

        assert not (
                np.isinf(sIn) or
                np.isnan(sIn) or
                np.isinf(sOut) or
                np.isnan(sOut) or
                np.isinf(g[x[0]]) or
                np.isnan(g[x[0]]))

        for i in range(1, n):
            sIn -= expw[x[i - 1]]
            sOut += 1.0 / sIn
            g[x[i]] = 1.0 - expw[x[i]] * sOut

            # if np.isinf(sIn) or np.isnan(sIn) or np.isinf(sOut) or np.isnan(sOut) or np.isinf(g[x[i]]) or np.isnan(
            #         g[x[i]]):
            #     return False

        return True  # If here, everything was ok

    @staticmethod
    def calc_w_log_F(U, w_log, fitnesses, delta_w_log_ps, nb_samples_lambda, ):
        # gradient = np.zeros_like(w_log)

        f_vals = U(fitnesses)  # list of scalar with len nb_samples_lambda
        assert len(f_vals) == nb_samples_lambda
        assert len(delta_w_log_ps) == nb_samples_lambda

        # old way, slow
        # for i in range(nb_samples_lambda):
        #     gradient += f_vals[i] * delta_w_log_ps[i]  # scalar * vector

        gradient = np.dot(f_vals, delta_w_log_ps)  # f_vals is a vector, delta_w_log_ps is a matrix
        # f_vals[i] will multiply the i'th row of delta_w_log_ps, then sum over all rows, somehow it works

        gradient /= nb_samples_lambda  # ??? maybe this is wrong? i think they dont do it in the code

        return gradient


if __name__ == "__main__":
    # Example usage
    n = 5
    expw = np.random.rand(n)  # Replace this with your actual data
    x = np.random.permutation(n)  # Example ranking
    g = np.zeros_like(expw, dtype=float)

    success = PlackettLuce.grad_log_prob(g, x, expw)

    grad2 = PlackettLuce.calc_w_log_ps(np.log(expw), np.array([x]))

    assert np.allclose(g, grad2[0])

    if success:
        print("Gradient computation successful.")
        print("Gradient values:", g)
    else:
        print("Error encountered during gradient computation.")

    # # test calc_w_log_ps
    # w_log = np.array([1, 2, 3])
    # sigmas = np.array([[0, 1, 2], [1, 0, 2]])
    # gradients = PlackettLuce.calc_w_log_ps(w_log, sigmas)
