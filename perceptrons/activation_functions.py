
import math
import sys

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

    # Eficiencia: evitamos hacer el calculo si ya sabemos que tiende a 0 o 1
    if x > MAX_X_RANGE / (-2 * beta):
        return 0.999
    elif x < MAX_X_RANGE / (2 * beta):
        return 0.001

    return 1 / (1 + math.exp(-2 * x * beta))


def theta_logistic_derivative(beta, x):
    theta_result = theta_logistic(beta, x)
    return 2 * beta * theta_result * (1 - theta_result)


def theta_tanh(beta, x):
    return math.tanh(x * beta)


def theta_tanh_derivative(beta, x):
    theta_result = theta_tanh(x, beta)
    return beta * (1 - theta_result ** 2)

