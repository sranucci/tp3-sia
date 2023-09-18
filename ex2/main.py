import random

from perceptron import *
from linear_perceptron import *
from non_linear_perceptron import *
import csv


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
        next(reader)        # ignoramos los valores nombres de las columnas
        for row in reader:
            input_data.append(list(map(lambda x: float(x),row[:-1])))       # los datos de input nos los quedamos
            output_data.append(float(row[-1]))

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


main()
