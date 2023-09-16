from perceptron import *
from linear_perceptron import *
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

    min_weights = perceptron(input_data, output_data, 0.2, 0.1, update_weights_linear, error_linear, theta_linear, 10000000)

    print(min_weights)

main()