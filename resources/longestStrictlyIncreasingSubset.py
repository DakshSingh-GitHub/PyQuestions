from bisect import bisect_left

def longestStrictlyIncreasingSubset(arr: list) -> int:
    if not arr:
        return 0
    tails = []
    for x in arr:
        idx = bisect_left(tails, x)
        if idx < len(tails):
            tails[idx] = x
        else:
            tails.append(x)
    return len(tails)
