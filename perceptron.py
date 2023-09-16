import copy
import json
import sys
from random import randint
import numpy as np
from datetime import datetime


def convert_input(data_input):
    converted_array = []
    for elem in data_input:
        elem.insert(0, 1)
        converted = np.array(elem)
        converted_array.append(converted)
    return converted_array


def initialize_weights(data_size):
    weights = []
    for _ in range(data_size):
        weights.append(np.random.uniform(-1, 1))  # TODO check
    return np.array(weights)


def generate_results(data_input, weights, theta):
    generated_results = []
    for elem in data_input:
        generated_results.append(theta(np.dot(elem, weights)))
    return generated_results


def compute_activation(x_vector, weights, theta):
    return theta(np.dot(x_vector, weights))


def perceptron(data_input, data_output, learning_constant, epsilon, update_weights, compute_error, theta, limit=100000):
    data_size = len(data_input[0])
    iterations = 0
    weights = initialize_weights(data_size + 1)  # tenemos el w0 tambien
    min_error = float("inf")
    w_min = []
    metrics = {}
    metrics["weights"] = []
    collect_metrics(metrics, weights)

    converted_input = convert_input(data_input)

    while min_error > epsilon and iterations < limit:
        idx = randint(0, data_size - 1)

        x_vector = converted_input[idx]  # convert x from array to numpy (to make vectorial operations)

        activation = compute_activation(x_vector, weights, theta)

        update_weights(learning_constant, activation, converted_input[idx], data_output[idx], weights)

        error = compute_error(converted_input, weights, data_output, theta)

        collect_metrics(metrics, weights)

        if error < min_error:
            min_error = error
            w_min = copy.copy(weights)

        iterations += 1

    print(f"min error {min_error} and epsilon {epsilon}")
    print(f"iterations {iterations} and limit {limit}")
    export_metrics(metrics)

    return w_min


def collect_metrics(metrics, weights):
    metrics["weights"].append(weights.tolist())


def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"../results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))



