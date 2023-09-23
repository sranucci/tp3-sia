import copy
import math

import numpy as np
import random


def theta_logistic(x, beta):
    # Evitamos el overflow
    try:
        a = 1 + math.exp(-2 * x * beta)
    except OverflowError:
        a = float("inf")
    return 1 / a


def theta_logistic_derivative(x, beta):
    theta_result = theta_logistic(x, beta)
    return 2 * 1 * theta_result * (1 - theta_result)


class NeuronLayer:
    def __init__(self, previous_layer_neuron_amount, current_layer_neurons_amount, activation_function, lower_weight,
                 upper_weight):
        self.excitement = None
        self.output = None
        self.activation_function = activation_function

        # Se genera una matrix de (current_layer_neurons_amount x previous_layer_neuron_amount)
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(lower_weight, upper_weight))
        self.weights = np.array(weights)

    def compute_activation(self, prev_input):

        self.excitement = np.dot(self.weights, prev_input)  # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation

        self.output = self.activation_function(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo


class MultiPerceptron:
    def __init__(self, num_entry_neurons, num_hidden_layers, neurons_per_layer, num_output_neurons, activation_function,
                 derivative_activation_function, learning_constant, beta):

        if num_hidden_layers <= 0 or neurons_per_layer <= 0:
            raise ValueError("There must be at least 1 intermediary layer and 1 neuron per layer.")

        self.layers: [NeuronLayer] = []

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = math.log(1 / 0.98 - 1) / (-2 * beta)
        lower_weight = - upper_weight

        activation_function = np.vectorize(lambda x: activation_function(x, beta))

        # Creamos la primera capa
        self.layers.append(
            NeuronLayer(num_entry_neurons, num_entry_neurons, activation_function, lower_weight, upper_weight))

        # Creamos la primera capa interna
        self.layers.append(
            NeuronLayer(num_entry_neurons, neurons_per_layer, activation_function, lower_weight, upper_weight))

        # Creamos el resto de las capas interna
        for i in range(num_hidden_layers - 1):
            self.layers.append(
                NeuronLayer(neurons_per_layer, neurons_per_layer, activation_function, lower_weight, upper_weight))

        # Creamos la ultima capa
        self.layers.append(
            NeuronLayer(neurons_per_layer, num_output_neurons, activation_function, lower_weight, upper_weight))

        self.derivative_activation_function = np.vectorize(lambda x: derivative_activation_function(x, beta))
        self.learning_constant = learning_constant
        self.input = None

    def forward_propagation(self, input_data):
        current = np.array(input_data)
        self.input = current
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current

    def update_weights(self, delta_w):  # [matriz1,matriz2,matriz3]
        for idx, layer in enumerate(self.layers):
            layer.weights += delta_w[idx]

    def compute_error(self, data_input, expected_outputs):

        error_vector = []

        for idx, input in enumerate(data_input):
            output_result = self.forward_propagation(input)
            error_vector.append(np.power(expected_outputs[idx] - output_result, 2))

        total = 0
        for elem in error_vector:
            total += sum(elem)

        return 0.5 * total

    def back_propagation(self, expected_output, generated_output) -> list:
        delta_w = []

        # Calculamos el delta W de la capa de salida
        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(
            self.layers[-1].excitement)
        delta_w.append(
            self.learning_constant * prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(
                self.layers[idx].excitement)
            delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(
                self.layers[idx - 1].output.reshape(-1, 1)))
            prev_delta = delta

        # Calculamos el delta W de la capa inicial
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(
            self.layers[0].excitement)
        delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

    def train(self, epsilon, limit, input_data, expected_output, batch_rate=1):
        size = len(input_data)
        if size < batch_rate:
            raise ValueError("Batch size is greater than size of input.")

        i = 0
        error = None
        min_error = float("inf")
        w_min = None
        while min_error > epsilon and i < limit:
            c = 0
            if batch_rate == size:  # entire input data in one batch
                for idx, input in enumerate(input_data):
                    result = self.forward_propagation(input)
                    delta_w_matrix = self.back_propagation(expected_output[idx], result)
                    c += delta_w_matrix
            else:  # se eligen batch_rate random input values
                for _ in range(batch_rate):
                    number = random.randint(0, size - 1)
                    result = self.forward_propagation(input_data[number])
                    delta_w_matrix = self.back_propagation(expected_output[number], result)
                    c += delta_w_matrix

            self.update_weights(c)
            error = self.compute_error(np.array(input_data), np.array(expected_output))
            if error < min_error:
                min_error = error
                w_min = self.get_weights()
            i += 1

        return error, w_min

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(copy.deepcopy(layer.weights))
        return weights

    def test(self, input_test_data, expected_output):
        true_positive = 0
        true_negative = 0
        false_postive = 0
        false_negative = 0
        for idx, input in enumerate(input_test_data):
            result = self.forward_propagation(input)
            if result == expected_output[idx]:
                print("hola")