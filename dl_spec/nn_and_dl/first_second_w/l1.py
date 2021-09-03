import math
import numpy as np


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    # loss = [yhat[i] + y[i] for i in yhat for i in y]
    loss = abs(y - yhat)
    ### END CODE HERE ###

    return sum(loss)

# yhat = np.array([.9, 0.2, 0.1, .4, .9])
# y = np.array([1, 0, 0, 1, 1])
# print(L1(yhat, y))
# print("L1 = " + str(L1(yhat, y)))  # 1.1
