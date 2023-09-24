from perceptrons.single_perceptron import generate_results


def update_weights_simple(learning_constant, activation, data_input, data_expected_output, weights, theta_derivative, beta):
    if data_expected_output != activation:
        weights += learning_constant * (data_expected_output - activation) * data_input


# Activation function
def theta_simple(beta, x):
    return 1 if x >= 0 else -1


# --------------------- error calculations --------------------------

def accuracy_error_simple(data_input, weights, data_output, theta, beta):
    generated_results = generate_results(data_input, weights, theta, beta)

    count = 0

    for generated, expected in zip(generated_results, data_output):
        if generated == expected:
            count += 1

    return 1 - count / len(generated_results) # si accuracy es 100% => 1 - 1 => 0


def module_error_simple(data_input, weights, data_output, theta, beta):
    generated_results = generate_results(data_input, weights, theta, beta)

    diff = 0
    for generated, expected in zip(generated_results, data_output):
        diff += abs(generated - expected)

    return diff

# ----------------------------------------------------------------


def collect_metrics(metrics, weights, error, iteration):
    metrics["weights"].append(weights.tolist())
    metrics["error"].append(error)
    metrics["iteration"] = iteration
