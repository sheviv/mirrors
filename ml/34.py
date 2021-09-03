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
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        d = {'type': 0, 'color': 1, 'name': 2}
        return sum(1 for item in items if item[d[ruleKey]] == ruleValue)


num1 = [["qqqq", "qqqq", "qqqq"], ["qqqq", "qqqq", "qqqq"], ["qqqq", "qqqq", "qqqq"],
        ["qqqq", "qqqq", "qqqq"], ["qqqq", "qqqq", "qqqq"], ["qqqq", "qqqq", "qqqq"],
        ["qqqq", "qqqq", "qqqq"]]
ruleKey = "name"
ruleValue = "qqqq"
s = Solution()
cv = s.countMatches(items=num1, ruleKey=ruleKey, ruleValue=ruleValue)
print(cv)
