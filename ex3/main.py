import random

from perceptrons.multi_perceptron import MultiPerceptron
import ex2.non_linear_perceptron as nlp
import numpy as np


def theta(x):
    return x


# def initialize_weights(layer_amounts):
#     weights_matrix_array = []
#     for idx, elem in enumerate(layer_amounts):
#         if idx == 0:
#             weights_matrix_array.append(np.random.rand(elem, elem))
#         else:
#             weights_matrix_array.append(np.random.rand(layer_amounts[idx-1], elem))
#
#     return weights_matrix_array


# -------------------------------------datos ejercicio 1: xor data------------------------
input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
expected_output_xor = [1, 1, -1, -1]
# ------------------------------------------------------------------------------


epsilon = 0.001
limit = 24
i = 0
# w = initialize_weights([4,5,5,3])
p = MultiPerceptron(2, 3, 5, 1, lambda x: 1 if x >= 0 else -1, lambda x: 1,
                    0.5)

error = None
min_error = float("inf")
w_min = None
while min_error > epsilon and i < limit:
    number = random.randint(0, len(input_data) - 1)
    result = p.forward_propagation(input_data[number])
    delta_w_matrix = p.back_propagation(expected_output_xor[number], result)
    p.update_weights(delta_w_matrix)
    error = p.compute_error(np.array(expected_output_xor[number]))
    if error < min_error:
        min_error = error
        #
    i += 1

print(min_error, i)
