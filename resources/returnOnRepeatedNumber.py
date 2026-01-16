
arr: list = [1, 2, 3, 4, 5, 1, 6, 7, 2]

def returnOnRepeatedNumber(array: list):
	test_arr = set()
	for i in array:
		if i in test_arr:
			return i
		test_arr.add(i)
	return -1

print(returnOnRepeatedNumber(arr))
