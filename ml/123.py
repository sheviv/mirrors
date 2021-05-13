import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from collections import Counter as mset
from itertools import groupby

class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        s_1 = "".join(word1)
        s_2 = "".join(word2)
        if s_1 == s_2:
            return True
        else:
            return False

word1 = ["abc", "d", "defg"]
word2 = ["abcddefg"]
s = Solution()
cv = s.arrayStringsAreEqual(word1=word1, word2=word2)
print(cv)

