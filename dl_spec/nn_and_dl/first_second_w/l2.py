import math
import numpy as np


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    loss = (y - yhat)**2
    ### END CODE HERE ###

    return sum(loss)

# yhat = np.array([.9, 0.2, 0.1, .4, .9])
# y = np.array([1, 0, 0, 1, 1])
# # print("L2 = " + str(L2(yhat,y)))
# print(L2(yhat, y))  # 0.43
