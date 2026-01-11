from typing import List

arr1: List = [1, 3, 11]
arr2: List = [2, 4, 7]


def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
	res_arr = sorted(nums1 + nums2)
	print(res_arr)
	if len(res_arr) % 2 == 0:
		return float(res_arr[int(len(res_arr) / 2) - 1] + res_arr[int(len(res_arr) / 2)]) / 2
	return float(res_arr[len(res_arr) // 2])

print(findMedianSortedArrays(arr1, arr2))
