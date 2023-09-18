from perceptron import perceptron
from simple_perceptron import *
import os
import json
import matplotlib.pyplot as plt
import numpy as np


def average_accuracy():
    LIMIT = 100
    error_each_iteration = [[] for _ in range(LIMIT+1)]
    for _ in range(100):
        input_data_and = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
        expected_output_and = [-1, -1, 1, -1]

        w_min, metrics = perceptron(input_data_and, expected_output_and, 0.8, 0, update_weights_simple, accuracy_error_simple,
                           theta_simple, collect_metrics, LIMIT)

        for idx in range(len(metrics["error"])):
            error_each_iteration[idx].append((1-metrics["error"][idx])*100)

    std_array = []
    mean_array = []

    for idx in range(LIMIT+1):
        mean_array.append(np.mean(error_each_iteration[idx]))
        std_array.append(np.std(error_each_iteration[idx]))

    mean_array = mean_array[1:]
    std_array = std_array[1:]

    #Create an array of x-values for the bars (e.g., categories or labels)
    x_values = np.arange(len(mean_array))

    # Create the bar chart with error bars
    plt.bar(x_values, mean_array, yerr=std_array, capsize=10, color='b', alpha=0.7)

    # Customize the plot
    # plt.xticks(x_values, ['Category 1', 'Category 2', 'Category 3', 'Category 4'])  # Replace labels as needed
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Mean accuracy for each iteration')

    # Show the plot
    plt.show()

average_accuracy()



