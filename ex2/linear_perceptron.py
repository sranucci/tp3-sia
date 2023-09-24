from perceptrons.single_perceptron import compute_activation


def theta_linear(beta, x):
    return x


def update_weights_linear(learning_constant, generated_output, data_input, expected_output, weights, theta_derivative, beta):
    if expected_output != generated_output:
        weights += learning_constant * (expected_output - generated_output) * data_input


def error_linear(converted_input, weights, data_output, theta, beta): # E(w)
    total = 0
    prev_sum = 0
    try:
        for data_input, expected_output in zip(converted_input, data_output):
            prev_sum = total
            generated = compute_activation(data_input, weights, theta, beta)
            total += (expected_output - generated) ** 2
    except OverflowError:
        print(prev_sum)
    return total / 2