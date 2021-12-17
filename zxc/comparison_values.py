nums = [3,2,1,5,4]
k = 2
c = 0
for idx, i in enumerate(nums):
    if idx != len(nums) - 1:
        for j in nums[idx + 1:]:
            # print(f"i: {i}, j: {j}")
            if abs(i - j) == k:
                c += 1
print(f"c: {c}")