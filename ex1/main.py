from perceptron import perceptron
from simple_perceptron import *
from animation import animate_lines



def main():
    input_data_and = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_and = [-1, -1, 1, -1]

    print("Running \'AND\' data set")
    w_min = perceptron(input_data_and,expected_output_and, 0.8, 0, update_weights_simple, module_error_simple, theta_simple, 10000000)

    print(w_min)

    input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    animate_lines(input_data, "and_gif")

    input_data_xor = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_xor = [1, 1, -1, -1]

    print("Running \'XOR\' data set")
    w_min_xor = perceptron(input_data_xor, expected_output_xor, 0.8, 0, update_weights_simple, module_error_simple, theta_simple, 40)

    print(w_min_xor)

    animate_lines(input_data, "xor_gif")


main()



