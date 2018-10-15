import sys

nums = []
for line in sys.stdin:
    num = int(line)
    nums.append(num)

nums.sort()
for num in nums:
    print(num)
