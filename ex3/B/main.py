import json
from perceptrons.multi_perceptron import *
import plotly.graph_objects as go

INPUT_SIZE = 35
OUTPUT_SIZE = 2


def main():
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

    train_input, train_expected_output, test_input, test_expected_output, test_number_value = separate_train_test(input_data, expected_output)

    neuron_network = MultiPerceptron(
     INPUT_SIZE,
     config["hidden_layer_amount"],
     config["neurons_per_layer"],
     OUTPUT_SIZE,
     theta_logistic,
     theta_logistic_derivative,
     config["hidden_layer_amount"],
     config["activation_function"]["beta"]
    )

    error, w_min, metrics = neuron_network.train(
        config["epsilon"],
        config["limit"],
        config["optimization_method"]["alpha"],
        train_input,
        train_expected_output,
        collect_metrics,
        config["batch_size"]
    )

    if config["print_results"]:
        print(f"error: {error}")

        accuracy, precision, recall, f1_score = neuron_network.test(test_input, test_expected_output)
        print(f"accuracy: {accuracy}\nprecision: {precision}\nrecall: {recall}\nf1_score: {f1_score}\n")

    if config["generate_graph"]:
        results = []
        for test_value in test_input:
            results.append(neuron_network.forward_propagation(test_value))
        # Convert the lists to numpy arrays for easier manipulation
        results = np.array(results)
        test_expected_output = np.array(test_expected_output)
        test_number_value = np.array(test_number_value)

        # Define colors for each set of values
        colors = ['red', 'blue']

        # Create a scatter plot for results and expected values
        fig = go.Figure()
        for i in range(2):  # Two values in each set (0 and 1)
            fig.add_trace(go.Scatter(x=test_number_value, y=results[:, i], mode='markers', name=f'Results ({i})',
                                     marker=dict(color=colors[i])))
            fig.add_trace(go.Scatter(x=test_number_value, y=test_expected_output[:, i], mode='markers',
                                     name=f'Expected ({i})', marker=dict(symbol='x', color=colors[i])))

        # Set axis labels and title
        fig.update_layout(
            xaxis_title='Test Number Values',
            yaxis_title='Values',
            title='Scatter Plot of Results and Expected Outputs'
        )

        # Show the plot
        fig.show()



def collect_metrics(metrics, error, iteration):
    pass


def separate_train_test(input_data, expected_output):
    train_input = []
    train_expected = []
    test_input = []
    test_expected = []
    test_number_value = []
    for idx in range(len(input_data)):
        if idx < 7:
            train_input.append(input_data[idx])
            train_expected.append(expected_output[idx])
        else:
            test_expected.append(expected_output[idx])
            test_input.append(input_data[idx])
            test_number_value.append(idx)

    return train_input, train_expected, test_input, test_expected, test_number_value


main()
