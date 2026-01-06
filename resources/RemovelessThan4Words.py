
arr = ["apple", "bat", "cat", "dante", "rat"]

def removeLessThan4Words(arraylist: list):
	ret = []
	for i in arraylist:
		if len(i) > 4:
			ret.append(i.upper())
	return ret


arr = removeLessThan4Words(arr)
print(arr)

