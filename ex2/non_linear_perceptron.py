import math

from ex2.linear_perceptron import error_linear
from perceptrons.single_perceptron import compute_activation


def update_weights_non_linear(learning_constant, generated_output, data_input, expected_output, weights, theta_derivative, beta):
    if expected_output != generated_output:
        weights += learning_constant * (expected_output - generated_output) * compute_activation(data_input, weights, theta_derivative, beta) * data_input


def error_non_linear(converted_input, weights, data_output, theta, beta):  # E(w)
    return error_linear(converted_input, weights, data_output, theta, beta)
