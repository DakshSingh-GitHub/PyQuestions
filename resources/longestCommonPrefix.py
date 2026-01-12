
strs = ["flower","fly","florine"]

def longestCommonPrefix(str_l: str) -> str:
	longest_prefix = ""
	for i in zip(*str_l):
		if len(set(i)) == 1:
			longest_prefix += i[0]
		else:
			break
	return longest_prefix

print(longestCommonPrefix(strs))
print(type(longestCommonPrefix(strs)))

# first we are destructuring each element with zip(*str_l) as [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'y', 'o')]
# then we are changing each element to a set to see if it has duplicates, since a set never allows duplicates
# ('f', 'f', 'f') will be changed to {'f'}
# if it's length (len(iterator)) = 1 that is element is same on all the other elements of 'strs'
# so we add 'f' to longest_prefix. Now same for 'l', but
# for the third, ('o', 'y', 'o') -> {'o', 'y'} hence duplicates, length != 1 so, ignore
