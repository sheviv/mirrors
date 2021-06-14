from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from collections import Counter as mset
from itertools import groupby
from datetime import datetime
from statistics import mode


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        cv = []
        f = 0
        # for idx, i in enumerate(prices[f:]):
        for i in range(0, len(prices)):
            print(f"i: {i}")
            if i == len(prices):
                pass
            else:
                cv.append(prices[i] - max(prices[i + 1:]))
            # cv.append()
        return True


prices = [7, 1, 5, 3, 6, 4]
target = 2  # 4
k = 2
# pieces = [[2,4,6,8]]
s = Solution()
cv = s.maxProfit(prices=prices)
print(cv)
