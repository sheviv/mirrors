import math
import numpy as np

def sigmoid_derivative(x):
    u = np.zeros((1, 3))
    for i in x:
        s = 1 / (1 + math.exp(-i))
        ds = s * (1 - s)
        u[0][i - 1] = ds
    return u[0]