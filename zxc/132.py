import math
from typing import List
import numpy as np
from itertools import groupby
from math import sqrt
from collections import Counter
from collections import defaultdict
from itertools import permutations


class Solution:
    def countValidWords(self, sentence: str) -> int:
        vb = []
        c = 0
        cv = sentence.split()
        for i in cv:
            if any(char.isdigit() for char in i):
                pass
            else:
                if i.count("-") > 1:
                    pass
                elif i[0].isalpha():
                    if len(i) == 1 and i[len(i) - 1] == "!" or i[len(i) - 1] == "." or i[len(i) - 1] == "," or (i.count("-") <= 1 and i[0].isalpha() and i[len(i) - 1].isalpha()):
                        vb.append(i)
                        c += 1
                elif len(i) == 1 and i == "!" or i == "." or i == ",":
                    vb.append(i)
                    c += 1
        return c



# word1 = "cat and  dog"  # 3
# word1 = "!this  1-s b8d!"  # 0
# word1 = "alice and  bob are playing stone-game10"  # 5
# word1 = "he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."  # 6
# word1 = "!"  # 6
# word1 = "-"  # 6
# word1 = "a-b-c"  # 0
word1 = "a-!b"  # 0
# word1 = "ababc"  # [102,120,130,132,210,230,302,310,312,320]
c = Solution()
print(c.countValidWords(word1))
