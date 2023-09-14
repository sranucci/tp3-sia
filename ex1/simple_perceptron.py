import sys
from random import randint

import numpy as np


def simple_perceptron(data, limit=100000):
    data_size = get_size(data[0]["input"])
    iterations = 0
    weights = initialize_weights(data_size)
    min_error = sys.maxsize  # very high number
    learning_constant = 0.1  # TODO parametrize

    while min_error > 0 and iterations < limit:
        random_number = randint(0, data_size - 1)
        x_vector = np.array(data[random_number]["input"])  # convert x from array to numpy (to make vectorial operations)
        x_vector = np.insert(x_vector, 0, 1)  # add 1 at the beggining of the vector to calculate stuff in vectorial way

        activation = compute_activation(x_vector, weights)
        update_weights(learning_constant, activation, data[random_number], weights)

        error = compute_error(data, weights)
        if error < min_error:
            min_error = error
            w_min = weights

        iterations += 1


def compute_excitement(x_vector, weights):
    return np.dot(x_vector, weights)


def compute_activation(x_vector, weights):
    return 1 if compute_excitement(x_vector, weights) >= 0 else 0


def update_weights(learning_constant, activation, data, weights):
    weights += 2 * learning_constant * data["input"] * (data["expected_output"] - activation)


def compute_error(data, weights):
    return 1  # TODO


def initialize_weights(data_size):
    return np.zeros(data_size)


# amount of weights should be len(input) + 1
def get_size(data):
    return len(data) + 1
