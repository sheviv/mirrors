import math
import numpy as np

# GRADED
# FUNCTION: sigmoid


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = np.array([])
    for i in z:
        n = 1 / (1 + math.exp(-i))
        s = np.append(s, n)
    ### END CODE HERE ###

    return s
# print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))  # [ 0.5 0.88079708]
