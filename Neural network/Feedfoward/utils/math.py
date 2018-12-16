import numpy as np
import math



def sigmoid(x, derivate=False):
    # when derivate == True, it is required that x is the output of sigmoid
    # when derivate == False, it is required that x is the input of sigmoid
    return x * (1 - x) if derivate else 1 / (1 + np.exp(-x))

def tanh(x, derivate=False):
    return 1 - np.tanh(x)**2 if derivate else np.tanh(x)


def gelu(x):
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.pow(x, 3))))


def swish(x):
    return x * sigmoid(x)