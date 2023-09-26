import random

import plotly.express as px
import pandas as pd
import json
import plotly.figure_factory as ff
from ex3.C.main import apply_noise
from perceptrons.multi_perceptron import *
import plotly.graph_objects as go

OUTPUT_SIZE = 10
INPUT_SIZE = 35

def graph_mse_values_no_change(error_train, error_test):
    traces = []
    for idx, elem in enumerate(zip(error_train, error_test)):
        traces.append(go.Scatter(x=[i for i in range(len(elem[0]))], y=elem[0], mode='lines', name=f'EDS in training with {idx * 0.1} noise'))
        traces.append(go.Scatter(x=[i for i in range(len(elem[1]))], y=elem[1], mode='lines', name=f'EDS in testing with {idx * 0.1} noise'))


    # Create the layout for the plot
    layout = go.Layout(
        title='EDS for training and testing',
        xaxis=dict(title='Epochs'),
        yaxis=dict(title='EDS'),

    )

    # Create the figure and add traces
    fig = go.Figure(data=traces, layout=layout)

    fig.update_yaxes(range=[0, 3.5])

    # Show the plot
    fig.show()

def graph_mse_values(mse_values):
    # Create a trace for each configuration
    mse_values = [elem for elem in mse_values]

    traces = []
    for idx, mse in enumerate(mse_values):
        trace = go.Scatter(x=[i for i in range(len(mse))], y=mse, mode='lines', name=f'Neurons in hidden layer: {idx + 10}')
        traces.append(trace)

    # Create the layout for the plot
    layout = go.Layout(
        title='Training MSE Values vs Num. neurons in hidden layer',
        xaxis=dict(title='Neurons in hidden layer'),
        yaxis=dict(title='MSE'),
    )

    # Create the figure and add traces
    fig = go.Figure(data=traces, layout=layout)

    # Show the plot
    fig.show()

def graph_test_metrics(test_metrics):
    # Create a DataFrame
    # Define noise levels (replace with your noise levels)
    noise_levels = [i * 0.1 for i in range(len(test_metrics))]

    # Define metric names
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    # Create a DataFrame
    df = pd.DataFrame(test_metrics, columns=metrics)
    df['Noise Level'] = noise_levels

    # Create a grouped bar chart
    fig = go.Figure()

    for metric in metrics:
        fig.add_trace(go.Bar(
            x=df['Noise Level'],
            y=df[metric],
            name=metric
        ))

    # Customize the layout
    fig.update_layout(
        barmode='group',
        xaxis_title='Max Noise',
        yaxis_title='Score',
        title='Metrics vs. Max Noise',
        legend_title='Metrics'
    )

    # Show the plot
    fig.show()


def graph_confusion_matrix(matrix, i, j):
    confusion_matrix = matrix[i - 1][j - 1]
    class_labels = [i, j]

    fig = ff.create_annotated_heatmap(z=confusion_matrix,
                                      x=class_labels,
                                      y=class_labels,
                                      colorscale='Viridis')

    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted',
                      yaxis_title='Actual')

    # Add custom annotations to the cells for clarity (optional)
    annotations = []
    for i, row in enumerate(confusion_matrix):
        for j, val in enumerate(row):
            annotations.append(
                go.layout.Annotation(
                    x=class_labels[j],
                    y=class_labels[i],
                    text=str(val),
                    showarrow=False,
                    font=dict(color='white' if i == j else 'black')
                )
            )

    fig.update_layout(annotations=annotations)

    # Show the plot
    fig.show()


def collect_metrics(metrics, error, error_test, iteration):
    metrics["error"].append(error)
    metrics["error_test"].append(error_test)
    metrics["iterations"] = iteration

def average_train(data):
    config, train_input, train_output, test_input, test_output = data[0], data[1], data[2], data[3], data[4]
    errors = []
    errors_test = []
    for _ in range(10):
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

        error, w_min, metrics = neural_network.train(
            config["epsilon"],
            config["limit"],
            config["optimization_method"]["alpha"],
            np.array(train_input),
            np.array(train_output),
            collect_metrics,
            config["batch_size"]
        )
        errors.append(metrics["error"])
        errors_test.append(metrics["error_test"])

    sum_errors_train = np.zeros(len(errors[0]))

    for elem in errors:
        for idx, err in enumerate(elem):
            sum_errors_train[idx] += err

    for idx in range(len(sum_errors_train)):
        sum_errors_train[idx] /= len(errors)

    sum_errors_test = np.zeros(len(errors[0]))

    for elem in errors_test:
        for idx, err in enumerate(elem):
            sum_errors_test[idx] += err

    for idx in range(len(sum_errors_test)):
        sum_errors_test[idx] /= len(errors)

    return sum_errors_train, sum_errors_test


def average_test(data):
    config, input_data, expected_ouput, test_data, test_output = data[0], data[1], data[2], data[3], data[4]
    test_results = np.zeros(4)

    NUM_ITERATIONS = 10

    for _ in range(NUM_ITERATIONS):
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

        neural_network.train(
            config["epsilon"],
            config["limit"],
            config["optimization_method"]["alpha"],
            np.array(input_data),
            np.array(expected_ouput),
            collect_metrics,
            config["batch_size"]
        )

        accuracy, precision, recall, f1_score = neural_network.test(test_data, test_output)

        test_results[0] += accuracy
        test_results[1] += precision
        test_results[2] += recall
        if f1_score is not None:
            test_results[3] += f1_score
        else:
            test_results[3] += 0

    for idx in range(len(test_results)):
        test_results[idx] /= NUM_ITERATIONS

    return test_results


def average_matrix_confusion(data):
    config, input_data, expected_ouput, test_data, test_output = data[0], data[1], data[2], data[3], data[4]

    NUM_ITERATIONS = 50

    sum_matrix = None

    for _ in range(NUM_ITERATIONS):
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

        neural_network.train(
            config["epsilon"],
            config["limit"],
            config["optimization_method"]["alpha"],
            np.array(input_data),
            np.array(expected_ouput),
            collect_metrics,
            config["batch_size"]
        )

        confusion_matrix = neural_network.confusion_matrix(test_data, test_output)

        if sum_matrix is None:
            sum_matrix = confusion_matrix
        else:
            sum_matrix += confusion_matrix

    return sum_matrix




def average_train_parallel():
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

    train_error = []
    test_error = []

    combined = []
    for i, o in zip(input_data, expected_output):
        combined.append([i,o])

    random.shuffle(combined)
    input_data.clear()
    expected_output.clear()

    for i, o in combined:
        input_data.append(i)
        expected_output.append(o)

    TRAIN = 7

    train_input = input_data[:TRAIN]
    test_input = input_data[TRAIN:]

    train_output = expected_output[:TRAIN]
    test_output = expected_output[TRAIN:]

    data = []
    for i in range(3):
        data.append([config, train_input, train_output,
                     apply_noise(test_input, 0.1 * i),
                     test_output])

    with Pool(processes=3) as pool:
        results = pool.map(average_train, data)

        for elem in results:
            train_error.append(elem[0])
            test_error.append(elem[1])

    graph_mse_values_no_change(train_error, test_error)


def average_test_parallel():
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

    # combined = []
    # for i, o in zip(input_data, expected_output):
    #     combined.append([i, o])
    #
    # random.shuffle(combined)
    # input_data.clear()
    # expected_output.clear()
    #
    # for i, o in combined:
    #     input_data.append(i)
    #     expected_output.append(o)

    TRAIN = 7

    train_input = input_data
    test_input = input_data

    train_output = expected_output
    test_output = expected_output

    scores = []
    data = []
    for i in range(4):
        data.append([config, train_input, train_output, apply_noise(test_input, 0.1 * i), test_output])

    with Pool(processes=5) as pool:
        results = pool.map(average_test, data)

        for elem in results:
            scores.append(elem)

    graph_test_metrics(scores)

def average_test_confusion_parallel():
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

    train_input = input_data
    test_input = input_data

    train_output = expected_output
    test_output = expected_output

    scores = []
    data = []
    for i in range(4):
        data.append([config, train_input, train_output, apply_noise(test_input, 0.1 * i), test_output])

    with Pool(processes=5) as pool:
        results = pool.map(average_matrix_confusion, data)

        for elem in results:
            scores.append(elem)

    for matrix in scores:
        graph_confusion_matrix(matrix, 8,9)


if __name__ == "__main__":
    average_test_confusion_parallel()
