import json
from datetime import datetime
from perceptrons.multi_perceptron import *


OUTPUT_SIZE = 10
INPUT_SIZE = 35


def collect_metrics(metrics, error, iteration):
    metrics["error"].append(error)
    metrics["iterations"] = iteration


def export_metrics(metrics):
    now = datetime.now().strftime("%d-%m-%Y_%H%M%S")

    # Exporto las metricas
    with open(f"./results/results_{now}.json", mode="w+") as file:
        file.write(json.dumps(metrics, indent=4))


def apply_noise(inputs, max_noise):
    altered_inputs = []
    for number in inputs:
        altered = copy.deepcopy(number)
        for i in range(len(altered)):
            noise = random.uniform(0, max_noise)
            change = random.randint(0, 1)
            if change == 0:
                change = -1
            if altered[i] + change * noise > 1:
                altered[i] = 1
            elif altered[i] + change * noise < 0:
                altered[i] = 0
            else:
                altered[i] = altered[i] + change * noise
        altered_inputs.append(altered)

    return altered_inputs


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


    if config["selection_method"]["method"] == "simple":
        size = round(config["selection_method"]["proportion"] * len(input_data))
        training_input = input_data[:size]
        training_output = expected_output[:size]

        testing_input = input_data[size:]
        testing_output = expected_output[size:]

    elif config["selection_method"]["method"] == "none":
        training_input = input_data
        training_output = expected_output

        testing_input = input_data
        testing_output = expected_output
    else:
        raise ValueError("Invalid selection method")

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

    start_time = time.time()
    error, w_min, metrics = neural_network.train(
        config["epsilon"],
        config["limit"],
        config["optimization_method"]["alpha"],
        np.array(training_input),
        np.array(training_output),
        collect_metrics,
        config["batch_size"]
    )
    end_time = time.time()

    metrics["training error"] = error
    metrics["time elapsed"] = end_time - start_time

    accuracy, precision, recall, f1_score = neural_network.test(
        np.array(apply_noise(testing_input, config["max_noise"])),
        np.array(testing_output),
        0.05
    )

    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1 Score"] = f1_score

    export_metrics(metrics)




main()

