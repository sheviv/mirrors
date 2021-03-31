import math
import numpy as np


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + math.exp(-x))
    ### END CODE HERE ###

    return s
# print(basic_sigmoid(3))  # 0.9525741268224334