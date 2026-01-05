
arraySet = [2, 1, 6, 4, 5, 3]
targetNumber = 5

def twoSum(arr, target):
	out = []
	for i in range(len(arr)):
		for j in range(i+1, len(arr)):
			if arr[i] + arr[j] == target:
				setOut = (arr[i], arr[j])
				out.append(setOut)
	return out

print(twoSum(arraySet, targetNumber))
