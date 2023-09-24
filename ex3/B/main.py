import json

from perceptrons.activation_functions import theta_logistic, theta_logistic_derivative
from perceptrons.multi_perceptron import *

INPUT_SIZE = 35
OUTPUT_SIZE = 2


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

    neuron_network = MultiPerceptron(
     INPUT_SIZE,
     config["hidden_layer_amount"],
     config["neurons_per_layer"],
     OUTPUT_SIZE,
     theta_logistic,
     theta_logistic_derivative,
     config["hidden_layer_amount"],
     config["activation_function"]["beta"]
    )
    error, w_min = neuron_network.train(
        config["epsilon"],
        config["limit"],
        config["optimization_method"]["alpha"],
        input_data,
        expected_output,
        config["batch_size"]
    )
    print(f"error: {error}")

    for input, output in zip(input_data, expected_output):
        generated = neuron_network.forward_propagation(input)
        print(f"generated: {generated}, expected: {output}")


main()
