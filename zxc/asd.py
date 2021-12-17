# Function which returns subset or r length from n
from itertools import combinations
from itertools import permutations

# a = [0,1,2,3,4]
# cv = []
# asd = []
# s = ""
# for i in range(3, 4):
#     asd = list(combinations(a, i))
# print(f"asd: {asd}")



def dfg(zxc):
    cv = []
    perm = permutations(zxc, 3)
    for i in list(perm):
        num = "".join(list(map(str, i)))
        if num[0] != str(0) and int(num) % 2 == 0:
            cv.append(num)
    return cv

exx = [0,1,2,3,4]
# print(dfg(exx))


cv = []
word1 = [[1,2],[3,4],[5,6]]
# print(list(range(10, 100, 10)))
print(list(range(word1[0][0], word1[0][1] + 1, 1)))