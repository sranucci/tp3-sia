import math

from ex2.linear_perceptron import error_linear
from perceptron import compute_activation


def theta_tanh(x, beta):
    return math.tanh(x * beta)


def theta_tanh_derivative(x, beta):
    theta_result = theta_tanh(x, beta)
    return beta * (1 - theta_result ** 2)


def theta_logarithmic(x, beta):
    return 1 / (1 + math.exp(-2 * x * beta))


def theta_logaritmic_derivative(x, beta):
    theta_result = theta_logarithmic(x, beta)
    return 2 * beta * theta_result * (1 - theta_result)


def update_weights_linear(learning_constant, generated_output, data_input, expected_output, weights, theta_derivative, beta):
    if expected_output != generated_output:
        weights += learning_constant * (expected_output - generated_output) * compute_activation(data_input, weights, theta_derivative, beta) * data_input


def error_non_linear(converted_input, weights, data_output, theta, beta):  # E(w)
    return error_linear(converted_input, weights, data_output, theta, beta)
