from perceptrons.single_perceptron import perceptron
from simple_perceptron import *
import plotly.graph_objects as go
import numpy as np

def average_accuracy():
    LIMIT = 25
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

average_accuracy()