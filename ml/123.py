#!/usr/bin/python3
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
import datetime


class Solution:
    def backspaceCompare(self, S: str, T: str) -> bool:
        s, t = [], []
        for i in S:
            if i != '#':
                s = s + [i]
            else:
                s[:-1]
        for i in T:
            if i != '#':
                t = t + [i]
            else:
                t[:-1]
        return s == t
        # return True


nums = "ab#c"
t = "ad#c"
s = Solution()
cv = s.backspaceCompare(S=nums, T=t)
print(cv)
