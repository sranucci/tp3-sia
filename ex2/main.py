import random

from perceptron import *
from linear_perceptron import *
from non_linear_perceptron import *
import csv
from selection_methods import *


def run_algorithm(input_data, output_data, config):
    # Corremos los algoritmos
    if config["method"]["type"] == "linear":
        min_weights = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                 update_weights_linear, error_linear, theta_linear, config["limit"], 1)

    elif config["method"]["type"] == "non_linear":
        if config["method"]["theta"] == "tanh":
            min_weights = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                     update_weights_non_linear, error_non_linear, theta_logistic, config["limit"],
                                     config["method"]["beta"], theta_tanh_derivative)

        elif config["method"]["theta"] == "logistic":
            min_weights = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                     update_weights_non_linear, error_non_linear, theta_tanh, config["limit"],
                                     config["method"]["beta"], theta_logistic_derivative)
        else:
            quit("Invalid theta")
    else:
        quit("invalid method type")
    return min_weights


def test_weights(input_test_data, output_test_data, weights, config):
    # Generalizacion de modelo
    if config["method"]["type"] == "linear":
        error = generalization(input_test_data, output_test_data, weights, error_linear, theta_linear, 1)
    elif config["method"]["type"] == "non_linear":
        if config["method"]["theta"] == "tanh":
            error = generalization(input_test_data, output_test_data, weights, error_non_linear, theta_logistic,
                                   config["beta"])
        elif config["method"]["theta"] == "logistic":
            error = generalization(input_test_data, output_test_data, weights, error_non_linear, theta_logistic,
                                   config["beta"])
        else:
            quit("Invalid theta")
    else:
        quit("Invalid method type")

    return error


def main():
    with open("config.json") as file:
        config = json.load(file)

    if config["seed"] != -1:
        random.seed(config["seed"])

    input_data = []
    output_data = []

    # Obtenemos los valores
    with open(config["training_data"]) as file:
        reader = csv.reader(file)
        next(reader)  # ignoramos los valores nombres de las columnas
        for row in reader:
            input_data.append(list(map(lambda x: float(x), row[:-1])))  # los datos de input nos los quedamos
            output_data.append(float(row[-1]))

    # Seleccion de datos
    if config["selection_method"]["type"] == "simple":
        input_data, output_data, input_test_data, output_test_data = simple_selection(input_data, output_data,
                                                                                      config["selection_method"][
                                                                                          "proportion"])
        min_weights = run_algorithm(input_data, output_data, config)
        error = test_weights(input_test_data, output_test_data, min_weights, config)

    elif config["selection_method"]["type"] == "k-fold":
        input_data, output_data = k_fold(input_data, output_data, config["selection_method"]["folds"])
    else:
        quit("Invalid selection method")


main()
