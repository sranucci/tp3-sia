from perceptrons.multi_perceptron import MultiPerceptron


def theta(x):
    return x


p = MultiPerceptron(4, 2, 5, 3, theta)

p.forward_propagation([1, 2, 3, 4])


