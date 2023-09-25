import json
import time
import pandas as pd
from perceptrons.multi_perceptron import *
from perceptrons.selection_methods import simple_selection

OUTPUT_SIZE = 10
INPUT_SIZE = 35


def main():
    with open("./config.json") as file:
        config = json.load(file)

    if config["seed"] != -1:
        random.seed(config["seed"])

    # Levantamos el input
    input_data = []
    with open(config["training_data_input"], 'r') as file:
        temp = []
        for line in file:
            numbers = line.strip().split()

            temp.extend(map(int, numbers))

            if len(temp) == INPUT_SIZE:
                input_data.append(temp)
                temp = []

    if temp:
        input_data.append(temp)

    # Levantamos el output
    expected_output = []
    with open(config["training_data_output"], 'r') as file:
        for line in file:
            numbers = line.strip().split()
            arr = []
            for elem in numbers:
                arr.append(int(elem))
            expected_output.append(arr)


    neural_network = MultiPerceptron(
        INPUT_SIZE,
        config["hidden_layer_amount"],
        config["neurons_per_layer"],
        OUTPUT_SIZE,
        theta_logistic,
        theta_logistic_derivative,
        config["hidden_layer_amount"],
        config["activation_function_beta"],
    )

    start_time = time.time()
    error, w_min, metrics = neural_network.train(
        config["epsilon"],
        config["limit"],
        config["optimization_method"]["alpha"],
        np.array(input_data),
        np.array(expected_output),
        collect_metrics,
        config["batch_size"]
    )
    end_time = time.time()

    print(f"Training complete! \nError:{error}, Time elapsed:{end_time - start_time}s", )

    accuracy, precision, recall, f1_score = neural_network.test(np.array(input_data), np.array(expected_output))

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")


def collect_metrics(metrics, error, iteration):
    pass

main()

