import copy

from perceptrons.single_perceptron import perceptron
from simple_perceptron import *
import plotly.graph_objects as go
import numpy as np


def average_accuracy(limit, runs, run_and_data_set, learning_constant):
    error_each_iteration = [[] for _ in range(limit+1)]
    for _ in range(runs):
        input_data = [[-1, 1], [1, -1], [1, 1], [-1, -1]]
        expected_output_and = [-1, -1, 1, -1]
        expected_output_xor = [1, 1, -1, -1]

        w_min = None
        metrics = None

        if run_and_data_set:
            w_min, metrics = perceptron(copy.deepcopy(input_data), expected_output_and, learning_constant, 0, update_weights_simple, accuracy_error_simple,
                           theta_simple, collect_metrics, limit)
        else:
            w_min, metrics = perceptron(copy.deepcopy(input_data), expected_output_xor, learning_constant, 0, update_weights_simple, accuracy_error_simple,
                            theta_simple, collect_metrics, limit)

        for idx in range(len(metrics["error"])):
            error_each_iteration[idx].append((1-metrics["error"][idx])*100)

    std_array = []
    mean_array = []

    for idx in range(limit+1):
        mean_array.append(np.mean(error_each_iteration[idx]))
        std_array.append(np.std(error_each_iteration[idx]))

    mean_array = mean_array[1:]
    std_array = std_array[1:]

    # Create an array of x-values for the bars (e.g., categories or labels)
    x_values = np.arange(len(mean_array))

    # Create the bar chart with error bars using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=x_values,
            y=mean_array,
            error_y=dict(type='data', array=std_array, visible=True),
            marker=dict(color='blue', opacity=0.7)
        )
    ])

    # Customize the plot
    fig.update_layout(
        xaxis_title='Iteration',
        yaxis_title='Accuracy',
        title='Mean accuracy for each iteration'
    )

    # Show the interactive plot in a Jupyter Notebook or an HTML file
    fig.show()
