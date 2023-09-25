import json
import time
import pandas as pd
from perceptrons.multi_perceptron import *

OUTPUT_SIZE = 10
INPUT_SIZE = 784


def main():
    with open("./config.json") as file:
        config = json.load(file)

    if config["seed"] != -1:
        random.seed(config["seed"])

    data = pd.read_csv('../../training_data/mnist_train.csv')

    labels = data['label'].values
    pixels = data.drop('label', axis=1).values

    one_hot_labels = np.zeros((len(labels), OUTPUT_SIZE))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1

    pixel_arrays = np.array(pixels)
    expected_arrays = np.array(one_hot_labels)

    pixel_arrays = pixel_arrays[:1000]
    expected_arrays = expected_arrays[:1000]

    neuronNetwork = MultiPerceptron(
        INPUT_SIZE,
        config["hidden_layer_amount"],
        config["neurons_per_layer"],
        OUTPUT_SIZE,
        theta_logistic,
        theta_logistic_derivative,
        config["hidden_layer_amount"],
        config["activation_function"]["beta"],
        )

    start_time = time.time()
    print("starting training")
    error, w_min, metrics  = neuronNetwork.train(
        config["epsilon"],
        config["limit"],
        config["optimization_method"]["alpha"],
        pixel_arrays,
        expected_arrays,
        collect_metrics,
        config["batch_size"]
    )
    end_time = time.time()
    print(f"error:{error}, time:{end_time - start_time}s", )


def collect_metrics(metrics, error, iteration):
    pass

main()

