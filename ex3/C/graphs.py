import random

import plotly.express as px
import pandas as pd
import json
from datetime import datetime

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

def graph_test_results(test_results):
    # Create subplots
    fig = go.Figure()
    neuron_configurations = [10, 11, 12, 13, 14]
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for elem in test_results:
        accuracy_values.append(elem[0])
        precision_values.append(elem[1])
        recall_values.append(elem[2])
        f1_score_values.append(elem[3])

    performance_metrics = {
        'Accuracy': accuracy_values,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_score_values,
    }

    # Create a separate bar chart for each performance metric
    for metric_name, metric_values in performance_metrics.items():
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=neuron_configurations,
            y=metric_values,
            name=metric_name,
            marker_color='blue'  # You can customize the color here
        ))

        fig.update_layout(
            title=f'{metric_name} for Different Configurations',
            xaxis=dict(title='Neurons per Layer'),
            yaxis=dict(title=metric_name),
        )

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
            np.array(test_input),
            np.array(test_output),
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
    config, input_data, expected_ouput, test_data = data[0], data[1], data[2], data[3]
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

        accuracy, precision, recall, f1_score = neural_network.test(test_data, expected_ouput)

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

    test_data = apply_noise(input_data, 0.01)

    scores_per_change = []

    data = []
    for i in range(6):
        new_config = copy.deepcopy(config)
        new_config["neurons_per_layer"] += i
        data.append([new_config, input_data, expected_output, test_data])

    with Pool(processes=5) as pool:
        results = pool.imap_unordered(average_test, data)

        for elem in results:
            scores_per_change.append(elem)

    graph_test_results(scores_per_change)


if __name__ == "__main__":
    average_train_parallel()
