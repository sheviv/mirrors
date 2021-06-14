from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
from collections import Counter as mset
from itertools import groupby
from datetime import datetime


# class Solution:
#     def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
#
#         return True
#
#
#
# # nums = [3,2,3]
# numBottles = 9  # 13
# numExchange = 3
# s = Solution()
# cv = s.numWaterBottles(numBottles=numBottles, numExchange=numExchange)
# print(cv)

# class Solution:
#     # def hammingWeight(self, n: int) -> int:
#     def hammingWeight(self, n) -> int:
#         c = 0
#         cv = bin(n)
#         for i in cv:
#             if i == "1":
#                 c += 1
#         print(f"c: {c}")
#         return str(bin(n))[2:].count("1")
#
# pieces = "00000000000000000000000000001011"
# s = Solution()
# cv = s.hammingWeight(n=pieces)
# print(cv)


class Solution(object):
    def isHappy(self, n):
        return self.solve(n, {})

    def solve(self, n, visited):
        if n == 1:
            return True
        if n in visited:
            return False
        visited[n] = 1
        l = list(map(int, list(str(n))))
        temp = 0
        for i in l:
            temp += (i ** 2)
        return self.solve(temp, visited)


ob1 = Solution()
op = ob1.isHappy(19)
print("Is Happy:", op)
