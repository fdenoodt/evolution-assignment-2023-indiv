from abc import ABC, abstractmethod

import numpy as np

from utility import Utility


#
#
# if __name__ == "__main__":
#     # set seed
#     np.random.seed(123456)
#
#     # test sample_permutations
#     n = 5
#     nb_samples_lambda = 10
#
#     # Example vanilla pdf
#     # w = np.random.rand(n)
#     w = np.array([1, 110, 220, 455, 999])
#     pdf = VanillaPdf(n, np.log(w))
#     sigmas = pdf.sample_permutations(nb_samples_lambda)
#     sigmas2 = pdf.sample_permutations_slow(nb_samples_lambda)
#     print(sigmas)
#     print(sigmas2)
#
#     print("*" * 80)
#     # Example conditional pdf
#     pdf = ConditionalPdf(n)
#     sigmas = pdf.sample_permutations(nb_samples_lambda)
#     print(sigmas)
#
#     print("*" * 80)
#     print("Test conditional pdf with fixed w")
#     W = np.array([[1, 1, 1, 1],
#                   [1, 1, 1, 1000],
#                   [1, 100, 100, 100],
#                   [1, 1, 1, 1]])
#     n = W.shape[0]
#
#     w_log = np.log(W)
#     pdf = ConditionalPdf(n, w_log)
#     sigmas = pdf.sample_permutations(1)
#     print(sigmas)
#
#     print("*" * 80)
#     print("Test conditional pdf gradients")
#     gradient = pdf.calc_gradients(sigmas)
#     print(gradient)
#
#     print("*" * 80)
#     print("Test conditional pdf distinct permutations nodes")
#     n = 10
#     pdf = ConditionalPdf(n)
#     sigma = pdf.sample_permutation()
#     print(np.exp(pdf.w_log))
#     print(sigma)
#     assert len(np.unique(sigma)) == n
#
#     print("*" * 80)
#     print("Test np.sum")
#     w = np.array([[1, 2, 3],
#                   [1, 2, 3],
#                   [1, 2, 3]], dtype=float)
#     w /= np.sum(w, axis=0)  # normalize each column
#     print(w)
#
#     # Example usage
#     # n = 5
#     # expw = np.random.rand(n)  # Replace this with your actual data
#     # x = np.random.permutation(n)  # Example ranking
#     # g = np.zeros_like(expw, dtype=float)
#     #
#     # success = PlackettLuce.grad_log_prob(g, x, expw)
#     #
#     # grad2 = PlackettLuce.calc_w_log_ps(np.log(expw), np.array([x]))
#     #
#     # assert np.allclose(g, grad2[0])
#     #
#     # if success:
#     #     print("Gradient computation successful.")
#     #     print("Gradient values:", g)
#     # else:
#     #     print("Error encountered during gradient computation.")
#
#     # # test calc_w_log_ps
#     # w_log = np.array([1, 2, 3])
#     # sigmas = np.array([[0, 1, 2], [1, 0, 2]])
#     # gradients = PlackettLuce.calc_w_log_ps(w_log, sigmas)
#
#     # # test calc_gradients in PlackettLuce
#     # n = 5
#     # expw = np.random.rand(n)  # Replace this with your actual data
#     # x = np.random.permutation(n)  # Example ranking
#     # g = np.zeros_like(expw, dtype=float)
#     #
#     # pdf = VanillaPdf(n)
#     #
#     # gradients = pdf.calc_gradients(np.log(expw), np.array([x]))
#     # print(gradients)
