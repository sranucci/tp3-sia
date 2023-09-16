from perceptron import *
from linear_perceptron import *
from non_linear_perceptron import *
import csv


def main():
    input_data = []
    output_data = []
    with open("../training_data/ex2.csv") as file:
        reader = csv.reader(file)
        next(reader)        # ignoramos los valores nombres de las columnas
        for row in reader:
            input_data.append(list(map(lambda x: float(x),row[:-1])))       # los datos de input nos los quedamos
            output_data.append(float(row[-1]))

    # min_weights = perceptron(input_data, output_data, 0.5, 1000, update_weights_linear, error_linear, theta_linear,5000000, 1)

    min_weights_non_linear = perceptron(input_data, output_data, 0.5, 1000, update_weights_linear, error_non_linear, theta_logarithmic,1000000, 0.3, theta_logaritmic_derivative)

    print(min_weights_non_linear)


main()
