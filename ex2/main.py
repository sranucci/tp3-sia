import random

from perceptrons.single_perceptron import *
from linear_perceptron import *
from non_linear_perceptron import *
import csv
from perceptrons.selection_methods import *


def collect_metrics(metrics, weights, error, iterations):
    metrics["error"].append(error)
    metrics["iteration"] = iterations

def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"results/resultsABC_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))


def run_algorithm(input_data, output_data, config):
    # Corremos los algoritmos
    if config["method"]["type"] == "linear":
        min_weights, metrics = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                          update_weights_linear, error_linear, theta_linear, collect_metrics,
                                          config["limit"], 1)

    elif config["method"]["type"] == "non_linear":
        if config["method"]["theta"] == "tanh":
            min_weights, metrics = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                              update_weights_non_linear, error_non_linear, theta_logistic,
                                              collect_metrics,
                                              config["limit"], config["method"]["beta"], theta_tanh_derivative)

        elif config["method"]["theta"] == "logistic":
            min_weights, metrics = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                              update_weights_non_linear, error_non_linear, theta_tanh, collect_metrics,
                                              config["limit"], config["method"]["beta"], theta_logistic_derivative)
        else:
            quit("Invalid theta")
    else:
        quit("invalid method type")

    export_metrics(metrics)
    return min_weights


def test_weights(input_test_data, output_test_data, weights, config):
    # Generalizacion de modelo
    if config["method"]["type"] == "linear":
        error = generalization(input_test_data, output_test_data, weights, error_linear, theta_linear, 1)
    elif config["method"]["type"] == "non_linear":
        if config["method"]["theta"] == "tanh":
            error = generalization(input_test_data, output_test_data, weights, error_non_linear, theta_tanh,
                                   config["method"]["beta"])
        elif config["method"]["theta"] == "logistic":
            error = generalization(input_test_data, output_test_data, weights, error_non_linear, theta_logistic,
                                   config["method"]["beta"])
        else:
            quit("Invalid theta")
    else:
        quit("Invalid method type")

    return error


def run_algorithm_with_kfold(input_data, output_data, config):
    min_error = None
    for fold in range(len(input_data)):

        input_training_data = []
        output_training_data = []
        for item in range(len(input_data)):
            if item != fold:
                input_training_data.extend(input_data[item])
                output_training_data.extend(output_data[item])

        # TODO RANA: ver tenma de la conversion que esta tirando excepcion (resuelto con un learning rate bajo)
        min_weights = run_algorithm(input_training_data, output_training_data, config)
        # creamos los datos de testeo, todos menos el k_fold

        error = test_weights(input_data[fold], output_data[fold], min_weights, config)
        if min_error is None or error < min_error:
            min_error = error

    return min_error


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
        input_data, output_data = k_fold_selection(input_data, output_data, config["selection_method"]["folds"])

        error = run_algorithm_with_kfold(input_data, output_data, config)
    else:
        quit("Invalid selection method")


    print(error)


main()
