import copy
import math
import sys
from functools import partial

import numpy as np
import random


# MAX_XB nos permite decidir si la funcion math.exp(-2 * x * beta) da overflow.
# Resulta de resolver la inecuacion: math.exp(-2 * x * beta)  < MAX_FLOAT
# No solo nos permite evitar el overflow, sino que tambien es mas eficiente dado
# que en muchos casos evita hacer math.exp(...). Ej: para limit=100 pasa de 42 segundos
# a 36 segundos.
MAX_XB = math.floor(math.log(sys.float_info.max) / -2) + 2

# MAX_X_RANGE permite evitar hacer el cálculo de math.exp(-2 * x * beta), en los
# casos que sabemos que la respuesta va a dar o muy cercano a 1 o muy cercano a 0.
# Resulta de resolver la ecuacion: 1 / (1 + math.exp(-2 * x * beta)) = 0.999
# Tambien se usa que la funcion es impar.
# Ej: para limit=100 pasa de 33.88 segundos a 28.74 segundos

MAX_X_RANGE = math.log(1/0.999 - 1)


def theta_logistic(beta, x):
    # Evitamos el overflow
    if x < 0 and x * beta < MAX_XB:
        return 0        # 1/inf = 0

    # TODO: check si es seguro hacer esto
    # Eficiencia: evitamos hacer el calculo si ya sabemos que tiende a 0 o 1
    if x > MAX_X_RANGE / (-2 * beta):
        return 0.999
    elif x < MAX_X_RANGE / (2 * beta):
        return 0.001

    return 1 / (1 + math.exp(-2 * x * beta))


def theta_logistic_derivative(beta, x):
    theta_result = theta_logistic(beta, x)
    return 2 * beta * theta_result * (1 - theta_result)


def update_delta_w(delta_w, delta_w_matrix):
    if delta_w is None:
        delta_w = delta_w_matrix
    else:
        delta_w += delta_w_matrix
    return delta_w


def convert_data(data_input, data_output):
    new_input = []
    new_output = []

    for i, o in zip(data_input, data_output):
        new_input.append(np.array(i))
        new_output.append(np.array(o))

    return np.array(new_input), np.array(new_output)


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

        # guardamos el dot producto dado que lo vamos a usar aca y en el backpropagation
        self.excitement = np.dot(self.weights,prev_input)

        self.output = self.activation_function(self.excitement)

        return self.output  # Se ejecuta la funcion sobre cada elemento del arreglo

    def update_weights(self, delta_w):
        # TODO: optimization
        #si estamos en gradient descent, no se pasa nada al parametro prev_delta_w
        self.weights += delta_w

    def update_weights_with_optimization(self,delta_w,prev_delta_w):
        self.weights += delta_w + 0.07 * prev_delta_w

class MultiPerceptron:

    def __init__(self, num_entry_neurons, num_hidden_layers, neurons_per_layer, num_output_neurons, activation_function,
                 derivative_activation_function, learning_constant, beta):

        if num_hidden_layers <= 0 or neurons_per_layer <= 0:
            raise ValueError("There must be at least 1 intermediary layer and 1 neuron per layer.")

        self.layers: [NeuronLayer] = []

        # Caclculamos el rango de valores iniciales para los weights
        upper_weight = math.log(1 / 0.98 - 1) / (-2 * beta)
        lower_weight = - upper_weight

        activation_function = np.vectorize(partial(activation_function, beta))

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

        self.derivative_activation_function = np.vectorize(partial(derivative_activation_function, beta))
        self.learning_constant = learning_constant
        self.input = None

    def forward_propagation(self, input_data):
        current = input_data
        self.input = input_data
        for layer in self.layers:
            current = layer.compute_activation(current)

        return current


    #le pasamos un segundo parametro si queremos usar la optimizacion
    def update_all_weights(self, delta_w):  # [matriz1,matriz2,matriz3]
        for idx, layer in enumerate(self.layers):
            layer.update_weights(delta_w[idx])

    def update_all_weights_with_optimization(self,delta_w,prev_delta_w):
        for idx, layer in enumerate(self.layers):
            if prev_delta_w is None:
                layer.update_weights(delta_w[idx])
            else:
                layer.update_weights_with_optimization(delta_w[idx],prev_delta_w[idx])

    def compute_error(self, data_input, expected_outputs):

        error_vector = []

        for i, o in zip(data_input, expected_outputs):
            output_result = self.forward_propagation(i)
            error_vector.append(np.power(o - output_result, 2))

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
        w_min = None
        min_error = float("inf")

        # Convertimos los datos de entrada a Numpy Array (asi no lo tenemos que hacer mientras procesamos)
        converted_input, converted_output = convert_data(input_data, expected_output)

        while min_error > epsilon and i < limit:

            delta_w = None # delta_w de todas las neuronas del sistema

            # usamos todos los datos
            if batch_rate == size:
                for i, o in zip(converted_input, converted_output):

                    result = self.forward_propagation(i)
                    delta_w_matrix = self.back_propagation(o, result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # usamos un subconjunto
            else:
                for _ in range(batch_rate):
                    number = random.randint(0, size - 1)

                    result = self.forward_propagation(converted_input[number])
                    delta_w_matrix = self.back_propagation(converted_output[number], result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # actualizamos los
            self.update_all_weights(delta_w)

            error = self.compute_error(converted_input, converted_output)
            if error < min_error:
                min_error = error
                w_min = self.get_weights()
            i += 1

        return error, w_min


    def train_with_optimization(self,epsilon, limit, input_data, expected_output, batch_rate = 1):
        size = len(input_data)
        if size < batch_rate:
            raise ValueError("Batch size is greater than size of input.")

        i = 0
        error = None
        w_min = None
        prev_delta_w = None
        min_error = float("inf")

        # Convertimos los datos de entrada a Numpy Array (asi no lo tenemos que hacer mientras procesamos)
        converted_input, converted_output = convert_data(input_data, expected_output)

        while min_error > epsilon and i < limit:

            delta_w = None # delta_w de todas las neuronas del sistema

            # usamos todos los datos
            if batch_rate == size:
                for i, o in zip(converted_input, converted_output):

                    result = self.forward_propagation(i)
                    delta_w_matrix = self.back_propagation(o, result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # usamos un subconjunto
            else:
                for _ in range(batch_rate):
                    number = random.randint(0, size - 1)

                    result = self.forward_propagation(converted_input[number])
                    delta_w_matrix = self.back_propagation(converted_output[number], result)

                    delta_w = update_delta_w(delta_w, delta_w_matrix)

            # actualizamos los
            self.update_all_weights_with_optimization(delta_w, prev_delta_w)

            error = self.compute_error(converted_input, converted_output)
            if error < min_error:
                min_error = error
                w_min = self.get_weights()
            i += 1
            prev_delta_w = delta_w

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
