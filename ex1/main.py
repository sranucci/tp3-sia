from perceptron import perceptron
from simple_perceptron import *
from animation import animate_lines



def main():
    input_data_and = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_and = [-1, -1, 1, -1]

    input_data_xor = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_xor = [1, 1, -1, -1]

    w_min = perceptron(input_data_and,expected_output_and, 0.2, 0, update_weights_simple, module_error_simple, theta_simple, 10000000)

    print(w_min)

    input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]

    animate_lines(input_data)


main()



