import numpy as np
import random


class NeuronLayer:
    def __init__(self, previous_layer_neuron_amount, current_layer_neurons_amount, activation_function):
        self.activation_function = activation_function
        weights = []
        for i in range(current_layer_neurons_amount):
            weights.append([])
            for j in range(previous_layer_neuron_amount):
                weights[i].append(random.uniform(-1, 1))
        self.weights = np.array(weights)

    def compute_activation(self, prev_input):
        prev_input = np.transpose(prev_input)
        result = self.weights @ prev_input    # dot product of vectors

        # TODO: ver si se puede hacer mas eficiente
        output = [self.activation_function(elem) for elem in result]

        return np.array([output])


class MultiPerceptron:
    def __init__(self, num_entry_neurons, num_hidden_layers, neurons_per_layer, num_output_neurons,
                 activation_function):

        self.layers: [NeuronLayer] = []
        # Creamos la primera capa
        self.layers.append(NeuronLayer(num_entry_neurons, num_entry_neurons, activation_function))

        # Creamos la primera capa hidden
        self.layers.append(NeuronLayer(num_entry_neurons, neurons_per_layer, activation_function))

        # Creamos el resto de las capas hidden
        for i in range(num_hidden_layers - 1):
            self.layers.append(NeuronLayer(neurons_per_layer, neurons_per_layer, activation_function))

        # Creamos la ultima capa
        self.layers.append(NeuronLayer(neurons_per_layer, num_output_neurons, activation_function))

    def forward_propagation(self, input_data):
        current = np.array(input_data)
        current = np.transpose(current)

        for layer in self.layers:
            current = layer.compute_activation(current)

        return current
