from perceptron import compute_activation


def theta_linear(x, beta):
    return x


def update_weights_linear(learning_constant, generated_output, data_input, expected_output, weights, theta_derivative, beta):
    if expected_output != generated_output:
        weights += learning_constant * (expected_output - generated_output) * data_input


def error_linear(converted_input, weights, data_output, theta, beta): # E(w)
    sum = 0
    for data_input, output in zip(converted_input, data_output):
        sum += (output - compute_activation(data_input, weights, theta, beta)) ** 2
    return sum / 2