import time

from perceptrons.multi_perceptron import *
import json


def main():
    with open("config.json") as file:
        config = json.load(file)

    input_data_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [1, 1, 0, 0]

    neuronNetwork = MultiPerceptron(2,
                                    config["hidden_layer_amount"],
                                    config["neurons_per_layer"],
                                    1,
                                    theta_logistic,
                                    theta_logistic_derivative,
                                    config["learning_constant"],
                                    config["activation_function"]["beta"],
                                    )

    start_time = time.time()
    error, w_min = neuronNetwork.train(config["epsilon"], config["limit"], input_data_xor, expected_output, config["batch_rate"])
    end_time = time.time()
    print(error, end_time - start_time)
    print(neuronNetwork.forward_propagation([-1, 1]))
    print(neuronNetwork.forward_propagation([1, -1]))
    print(neuronNetwork.forward_propagation([1, 1]))
    print(neuronNetwork.forward_propagation([-1, -1]))

main()