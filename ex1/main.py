import copy
import json
from datetime import datetime

from perceptrons.single_perceptron import perceptron
from simple_perceptron import *
from animation import animate_lines
from perceptron_average import average_accuracy


def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    with open(f"./results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))


def main():
    with open("config.json") as file:
        config = json.load(file)

    input_data = config["input_data"]

    if config["and"]["run_and"]:
        expected_output_and = config["and"]["expected_output_and"]

        print("Running \'AND\' data set")
        w_min, metrics_and = perceptron(copy.deepcopy(input_data),
                                        expected_output_and,
                                        config["and"]["learning_constant"],
                                        0,
                                        update_weights_simple,
                                        module_error_simple,
                                        theta_simple,
                                        collect_metrics,
                                        config["and"]["limit"])
        if config["and"]["print_wmin"]:
            print(f"\'AND\' w min: {w_min}")

        if config["and"]["generate_gif"]:
            export_metrics(metrics_and)
            animate_lines(input_data, "and_gif")

    if config["xor"]["run_xor"]:
        expected_output_xor = config["xor"]["expected_output_xor"]

        print("Running \'XOR\' data set")
        w_min_xor, metrics_xor = perceptron(copy.deepcopy(input_data),
                                            expected_output_xor,
                                            config["xor"]["learning_constant"],
                                            0,
                                            update_weights_simple,
                                            module_error_simple,
                                            theta_simple,
                                            collect_metrics,
                                            config["xor"]["limit"])

        if config["xor"]["print_wmin"]:
            print(f"\'XOR\' w min: {w_min_xor}")

        if config["xor"]["generate_gif"]:
            export_metrics(metrics_xor)
            animate_lines(input_data, "xor_gif")

    if config["average"]["run_average"]:
        average_accuracy(config["average"]["limit"],
                         config["average"]["number_of_runs"],
                         config["average"]["run_and_data_set"],
                         config["average"]["learning_constant"])


main()



