from perceptrons.single_perceptron import compute_activation, convert_input


def test(test_input, test_output, weights, theta, beta):
    f = open("results/train_lineal.csv", "w+")
    converted_input = convert_input(test_input)

    for i, o in zip(converted_input, test_output):
        result = compute_activation(i, weights, theta, beta)
        print(f"{result}, {o}", file=f)

    f.close()