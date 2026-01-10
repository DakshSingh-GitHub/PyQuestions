
arraySet = [1, 4, 5, 3, 7, 2, 9, 1, 1, 1, 3, 3, 2]

def removeDuplicateFromSortedArray(arr:list):
	resultant_sorted = sorted(arr)
	out = []
	for i in resultant_sorted:
		if i not in out:
			out.append(i)
	return out

print(removeDuplicateFromSortedArray(arraySet))
