import numpy as np


#
#
# if __name__ == "__main__":
#     # fixed seed
#     np.random.seed(123456)
#     #
#     # print("*" * 20)
#     # print("Testing swap mutation")
#     #
#     n = 10
#     # popul_size = 5
#     # mutation_rate = 1
#     # popul = np.array([np.arange(n) for _ in range(popul_size)])
#     # print("popul:")
#     # print(popul)
#     # Variation.swap_mutation(popul, mutation_rate)
#     # print("after mutate:")
#     # print(popul)
#     #
#     # print("*" * 20)
#     # print("Testing inversion mutation")
#     # popul = np.array([np.arange(n) for _ in range(popul_size)])
#     # print("popul:")
#     # print(popul)
#     # Variation.inversion_mutation(popul, mutation_rate)
#     # print("after mutate:")
#     # print(popul)
#     #
#     # print("*" * 20)
#     # print("Testing scramble mutation")
#     # popul = np.array([np.arange(n) for _ in range(popul_size)])
#     # print("popul:")
#     # print(popul)
#     # Variation.scramble_mutation(popul, mutation_rate)
#     # print("after mutate:")
#     # print(popul)
#
#     print("*" * 20)
#     print("Testing order crossover")
#     # popul = np.array([np.arange(1, n) for _ in range(2)])
#     popul = np.array([np.arange(1, n), np.random.permutation(n - 1) + 1])
#     print("popul before:")
#     print(popul)
#     popul = Variation.order_crossover(popul)
#     print("popul after:")
#     print(popul)
