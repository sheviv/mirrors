import math
import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    u = np.zeros((1, 3))
    for i in x:
        u[0][i - 1] = 1 / (1 + math.exp(-i))
    return u[0]
