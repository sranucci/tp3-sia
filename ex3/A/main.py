from perceptrons.multi_perceptron import *
import json

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
# input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
# expected_output_xor = [1, 1, -1, -1]
# # ------------------------------------------------------------------------------
#
#
# epsilon = 0.001
# limit = 100000
# i = 0
# p = MultiPerceptron(2, 3, 5, 1, lambda x: 1 if x >= 0 else -1, lambda x: 1,
#                     0.1)
#
# error = None
# min_error = float("inf")
# w_min = None
# while min_error > epsilon and i < limit:
#     number = random.randint(0, len(input_data) - 1)
#     result = p.forward_propagation(input_data[number])
#     delta_w_matrix = p.back_propagation(expected_output_xor[number], result)
#     p.update_weights(delta_w_matrix)
#     error = p.compute_error(np.array(input_data), np.array(expected_output_xor))
#     if error < min_error:
#         min_error = error
#         #
#     i += 1
#
# print(min_error, i)
#
# print(p.forward_propagation([-1, 1]))
# print(p.forward_propagation([1, -1]))
# print(p.forward_propagation([1, 1]))
# print(p.forward_propagation([-1, -1]))



def main():
    with open("config.json") as file:
        config = json.load(file)

    input_data_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [1, 1, -1, -1]

    neuronNetwork = MultiPerceptron(2, 2, 5, 1)

    neuronNetwork.train(0, 100000, input_data_xor, expected_output, 4)

    print(neuronNetwork.forward_propagation([-1, 1]))
    print(neuronNetwork.forward_propagation([1, -1]))
    print(neuronNetwork.forward_propagation([1, 1]))
    print(neuronNetwork.forward_propagation([-1, -1]))

main()