import json
from perceptron import perceptron
from simple_perceptron import *


def main():
    input_data_and = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_and = [-1, -1, 1, -1]

    input_data_xor = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_xor = [1, 1, -1, -1]

    w_min = perceptron(input_data_and,expected_output_and, 0.2, 0, update_weights_simple, module_error_simple, theta_simple, 10000000)

    print(w_min)

main()