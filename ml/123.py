import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from collections import Counter as mset
from itertools import groupby
import operator


class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        x = 121
        print(f"1: {list(str(x))}")
        print(f"2: {list(str(x))[::-1]}")
        if "".join(list(str(x))) == "".join(list(str(x))[::-1]):
            return True
        else:
            return False
        # return True


st = "abbxxxxzzy"  # [427,286]
s = Solution()
cv = s.largeGroupPositions(s=st)
print(cv)
