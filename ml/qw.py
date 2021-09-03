from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List
from collections import Counter
from collections import Counter as mset
from itertools import groupby
from datetime import datetime

matrix = [
    [1, 3, 5, 7],
    [10, 11, 16, 20],
    [23, 30, 34, 50],
    [63, 77, 87, 89],
    [90, 94, 97, 99]
]

def searchMatrixR(matrix, target):
    if len(matrix) > 1:
        c_i = len(matrix) // 2
        if target > matrix[c_i][0]:
            return searchMatrixR(matrix[c_i:], target)
        elif target < matrix[c_i][0]:
            return searchMatrixR(matrix[:c_i], target)
        else:
            return True
    else:
        if len(matrix[0]) > 1:
            c_i = len(matrix[0]) // 2
            if target > matrix[0][c_i]:
                return searchMatrixR([matrix[0][c_i:]], target)
            elif target < matrix[0][c_i]:
                return searchMatrixR([matrix[0][:c_i]], target)
            else:
                return True
        else:
            if target == matrix[0][0]:
                return True
            else:
                return False
t0 = time.time()
print(searchMatrixR(matrix, 7))
t1 = time.time()
print(f"t1-t0: {t1-t0}")


def searchMatrix(matrix, target):
    while len(matrix) > 1:
        c_i = len(matrix) // 2
        if target > matrix[c_i][0]:
            matrix = matrix[c_i:]
        elif target < matrix[c_i][0]:
            matrix = matrix[:c_i]
        else:
            return True
    while len(matrix[0]) > 1:
        c_j = len(matrix[0]) // 2
        if target > matrix[0][c_j]:
            matrix[0] = matrix[0][c_j:]
        elif target < matrix[0][c_j]:
            matrix[0] = matrix[0][:c_j]
        else:
            return True
        if matrix[0][0] == target:
            return True
        else:
            return False
t2 = time.time()
print(searchMatrix(matrix, 7))
t3 = time.time()
print(f"t2-t3: {t2-t3}")
