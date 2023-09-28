import random

from perceptrons.single_perceptron import *
from linear_perceptron import *
from non_linear_perceptron import *
import csv
from traing import test
from perceptrons.selection_methods import *


def collect_metrics(metrics, weights, error, iterations):
    metrics["error"].append(error)
    metrics["iteration"] = iterations

def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))

def normalize_output(output, min_function, max_function):
    max_output = max(output)
    min_output = min(output)
    normalized = []
    for elem in output:
        normalized.append((elem - min_output) / (max_output - min_output) * (max_function - min_function) + min_function)

    return normalized
def normalize_input(data, min_function, max_function):
    if not data:
        return []

    # Transpose the data matrix to work with columns
    columns = list(zip(*data))

    normalized_data = []

    for column in columns:
        min_value = min(column)
        max_value = max(column)

        if min_value == max_value:
            # Avoid division by zero if all values are the same
            normalized_column = [0.0] * len(column)
        else:
            # Normalize using the custom function (tanh in this case)
            normalized_column = [np.tanh(min_function + ((x - min_value) / (max_value - min_value)) * (max_function - min_function)) for x in column]

        normalized_data.append(normalized_column)

    # Transpose the normalized data back to the original format
    normalized_data = list(zip(*normalized_data))

    return normalized_data


def run_algorithm(input_data, output_data, config):
    # Corremos los algoritmos
    if config["method"]["type"] == "linear":
        min_weights, metrics = perceptron(input_data, output_data, config["learning_constant"], config["epsilon"],
                                          update_weights_linear, error_linear, theta_linear, collect_metrics,
                                          config["limit"], 1)
        test(input_data, output_data, min_weights, theta_linear, config["method"]["beta"])
    elif config["method"]["type"] == "non_linear":
        if config["method"]["theta"] == "tanh":
            normalized = normalize_output(output_data, -1, 1)
            min_weights, metrics = perceptron(input_data, normalized, config["learning_constant"], config["epsilon"],
                                              update_weights_non_linear, error_non_linear, theta_tanh,
                                              collect_metrics,
                                              config["limit"], config["method"]["beta"], theta_tanh_derivative)
            test(input_data, normalized, min_weights, theta_linear, config["method"]["beta"])

        elif config["method"]["theta"] == "logistic":
            normalized = normalize_output(output_data, 0, 1)
            min_weights, metrics = perceptron(input_data, normalized, config["learning_constant"], config["epsilon"],
                                              update_weights_non_linear, error_non_linear, theta_logistic, collect_metrics,
                                              config["limit"], config["method"]["beta"], theta_logistic_derivative)
            test(input_data, normalized, min_weights, theta_linear, config["method"]["beta"])

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
            normalized = normalize_output(output_test_data, -1, 1)
            error = generalization(input_test_data, normalized, weights, error_non_linear, theta_tanh,
                                   config["method"]["beta"])
        elif config["method"]["theta"] == "logistic":
            normalized = normalize_output(output_test_data, 0, 1)
            error = generalization(input_test_data, normalized, weights, error_non_linear, theta_logistic,
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

    #f = open("results/test_data.csv", "w+")
    #outputs= generate_results(convert_input(input_test_data), min_weights, theta_linear , config["method"]["beta"])
    #for data_input, output in zip(outputs, output_data):
    #    # aca test
    #    print(f"{data_input},{output}", file=f)
    #f.close()
    print(error)
    print()

main()
