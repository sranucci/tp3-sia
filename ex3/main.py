import math
import random

from perceptrons.multi_perceptron import MultiPerceptron
import ex2.non_linear_perceptron as nlp
import numpy as np


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



# ----------------------------------------------------

def theta_logistic(x):
    try:
        a = math.exp(-2 * x * 1)
    except OverflowError:
        a = float("inf")
    return 1 / (1 + a)

def theta_logistic_derivative(x):
    theta_result = theta_logistic(x)
    return 2 * 1 * theta_result * (1 - theta_result)

# TODO: remove!!!!
random.seed(1)

# Initialize an empty list to store the result
input = []

# Open the text file for reading
with open('../training_data/ej3-digitos.txt', 'r') as file:
    # Initialize a temporary list to store each row of numbers
    temp = []

    # Read each line from the file
    for line in file:
        # Split the line into individual numbers
        numbers = line.strip().split()

        # Convert each number from string to integer and append to the temporary list
        temp.extend(map(int, numbers))

        # Check if the temporary list contains 35 numbers
        if len(temp) == 35:
            # Append the temporary list to the result and reset it
            input.append(temp)
            temp = []

# Check if there are any remaining numbers in the temporary list
if temp:
    input.append(temp)

expected_output = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

epsilon = 0.01
limit = 100000
i = 0
p = MultiPerceptron(35, 2, 5, 2, theta_logistic, theta_logistic_derivative, 0.25)

error = None
min_error = float("inf")
w_min = None
while min_error > epsilon and i < limit:
    number = random.randint(0, len(input) - 1)
    result = p.forward_propagation(input[number])
    delta_w_matrix = p.back_propagation(expected_output[number], result)
    p.update_weights(delta_w_matrix)
    error = p.compute_error(np.array(input), np.array(expected_output))
    if error < min_error:
        min_error = error
    i += 1

print(min_error, i)


for array, output in zip(input, expected_output):
    print(f"expected: {output}, generated: {p.forward_propagation(array)}")




