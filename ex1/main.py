import json
from simple_perceptron import simple_perceptron


def main():
    data = json.load(open("./training_data.json"))
    simple_perceptron(data["and_data"])

main()