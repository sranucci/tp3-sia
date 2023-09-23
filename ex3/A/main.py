from perceptrons.multi_perceptron import *
import json


def main():
    with open("config.json") as file:
        config = json.load(file)

    input_data_xor = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
    expected_output = [1, 1, -1, -1]

    neuronNetwork = MultiPerceptron(2,
                                    config["hidden_layer_amount"],
                                    config["neurons_per_layer"],
                                    1,
                                    theta_logistic,
                                    theta_logistic_derivative,
                                    config["learning_constant"],
                                    config["activation_function"]["beta"],
                                    )

    neuronNetwork.train(0, config["limit"], input_data_xor, expected_output, config["batch_rate"])

    print(neuronNetwork.forward_propagation([-1, 1]))
    print(neuronNetwork.forward_propagation([1, -1]))
    print(neuronNetwork.forward_propagation([1, 1]))
    print(neuronNetwork.forward_propagation([-1, -1]))

main()