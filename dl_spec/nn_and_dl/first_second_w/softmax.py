import math
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x_exp = np.exp(x - np.max(x, axis=-1)[..., None])
    x_sum = x_exp.sum(axis=-1)[..., None]
    return x_exp / x_sum
