import copy
import json
from datetime import datetime

from perceptron import perceptron
from simple_perceptron import *
from animation import animate_lines


def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"../results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))


def main():
    input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
    expected_output_and = [-1, -1, 1, -1]

    print("Running \'AND\' data set")
    w_min, metrics_and = perceptron(copy.deepcopy(input_data), expected_output_and, 0.8, 0, update_weights_simple, module_error_simple, theta_simple, collect_metrics, 100)

    export_metrics(metrics_and)
    animate_lines(input_data, "and_gif")

    expected_output_xor = [1, 1, -1, -1]

    print("Running \'XOR\' data set")
    w_min_xor, metrics_xor = perceptron(copy.deepcopy(input_data), expected_output_xor, 0.8, 0, update_weights_simple, module_error_simple, theta_simple, collect_metrics, 40)

    export_metrics(metrics_xor)
    animate_lines(input_data, "xor_gif")


main()



