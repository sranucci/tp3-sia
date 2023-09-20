import copy
import json
from random import randint
import numpy as np
from datetime import datetime


def convert_input(data_input):
    converted_array = []
    for elem in data_input:
        convertible = [1, *elem]
        converted = np.array(convertible)
        converted_array.append(converted)
    return converted_array


def initialize_weights(data_size):
    weights = []
    for _ in range(data_size):
        weights.append(np.random.uniform(-10, 10))  # TODO check
    return np.array(weights)


def generate_results(data_input, weights, theta, beta):
    generated_results = []
    for elem in data_input:
        generated_results.append(theta(np.dot(elem, weights), beta))
    return generated_results


def compute_activation(x_vector, weights, theta, beta):
    return theta(np.dot(x_vector, weights), beta)


def perceptron(data_input, data_output, learning_constant, epsilon, update_weights, compute_error, theta,
               collect_metrics, limit=100000, beta=1, theta_derivative=None):
    data_size = len(data_input[0])
    iterations = 0
    weights = initialize_weights(data_size + 1)  # tenemos el w0 tambien
    min_error = float("inf")
    w_min = []
    metrics = {}
    initialize_metrics(metrics)
    collect_metrics(metrics, weights, -1, 0)

    converted_input = convert_input(data_input)

    while min_error > epsilon and iterations < limit:
        idx = randint(0, data_size - 1)

        x_vector = converted_input[idx]  # convert x from array to numpy (to make vectorial operations)
        activation = compute_activation(x_vector, weights, theta, beta)

        update_weights(learning_constant, activation, converted_input[idx], data_output[idx], weights, theta_derivative,
                       beta)

        error = compute_error(converted_input, weights, data_output, theta, beta)

        collect_metrics(metrics, weights, error, iterations + 1)

        if error < min_error:
            min_error = error
            w_min = copy.copy(weights)

        iterations += 1

    return w_min, metrics


def initialize_metrics(metrics):
    metrics["error"] = []
    metrics["weights"] = []
    metrics["iteration"] = 0


def generalization(test_input, test_output, weights, compute_error, theta, beta):
    converted_input = convert_input(test_input)
    return compute_error(converted_input, weights, test_output, theta, beta)
