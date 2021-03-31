import math
import numpy as np
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    length, height, depth = image.shape
    # print(length, height, depth)
    ### END CODE HERE ###

    return image.reshape((length * height * depth, 1))
