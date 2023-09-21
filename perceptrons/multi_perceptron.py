import numpy as np
import random


class NeuronLayer:
    def __init__(self, previous_layer_neuron_amount, current_layer_neurons_amount, activation_function):
        self.output_dot = None
        self.excitement = None
        self.output = None
        self.activation_function = activation_function

        # Se genera una matrix de (current_layer_neurons_amount x previous_layer_neuron_amount)
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(-100, 100))
        self.weights = np.array(weights)

    def compute_activation(self, prev_input):

        self.excitement = np.dot(self.weights, prev_input)  # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation

        self.output = self.activation_function(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo


class MultiPerceptron:
    def __init__(self, num_entry_neurons, num_hidden_layers, neurons_per_layer, num_output_neurons, activation_function,
                 derivative_activation_function, learning_constant):
        self.layers: [NeuronLayer] = []

        activation_function = np.vectorize(activation_function)

        # Creamos la primera capa
        self.layers.append(NeuronLayer(num_entry_neurons, num_entry_neurons, activation_function))

        # Creamos la primera capa interna
        self.layers.append(NeuronLayer(num_entry_neurons, neurons_per_layer, activation_function))

        # Creamos el resto de las capas interna
        for i in range(num_hidden_layers - 1):
            self.layers.append(NeuronLayer(neurons_per_layer, neurons_per_layer, activation_function))

        # Creamos la ultima capa
        self.layers.append(NeuronLayer(neurons_per_layer, num_output_neurons, activation_function))

        self.derivative_activation_function = np.vectorize(derivative_activation_function)
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



    def compute_error(self, expected_outputs):
        output = self.layers[-1].output

        error_vector = np.power(expected_outputs - output, 2)

        return 0.5 * sum(error_vector)

    def back_propagation(self, expected_output, generated_output) -> list:
        delta_w = []

        # Calculamos el delta W de la capa de salida
        prev_delta = (expected_output - generated_output) * self.derivative_activation_function(self.layers[-1].excitement)
        delta_w.append(self.learning_constant * prev_delta.reshape(-1, 1) @ np.transpose(self.layers[-2].output.reshape(-1, 1)))

        # Calculamos el delta W de las capas ocultas
        for idx in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(prev_delta, self.layers[idx + 1].weights) * self.derivative_activation_function(self.layers[idx].excitement)
            delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.layers[idx-1].output.reshape(-1, 1)))
            prev_delta = delta

        # Calculamos el delta W de la capa inicial
        delta = np.dot(prev_delta, self.layers[1].weights) * self.derivative_activation_function(self.layers[0].excitement)
        delta_w.append(self.learning_constant * delta.reshape(-1, 1) @ np.transpose(self.input.reshape(-1, 1)))

        delta_w.reverse()

        return delta_w

