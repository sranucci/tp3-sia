from perceptron import compute_activation


def theta_linear(x):
    return x


def update_weights_linear(learning_constant, activation, data_input, data_expected_output, weights):
    if data_expected_output != activation:
        weights += learning_constant * (data_expected_output - activation) * data_input


def error_linear(converted_input, weights, data_output, theta): # E(w)
    sum = 0
    for data_input, output in zip(converted_input, data_output):
        sum += (output - compute_activation(data_input, weights, theta)) ** 2
    return sum / 2