# Longest substring without repeating character

stringL = "Daksh Singh"

def longestSubString(string_test:str):
	string = string_test.lower()
	greatest = 0
	for i in string:
		temp = string.split(i)
		res = []
		for iIn in temp:
			res.append(len(iIn))
		max_l = max(res)
		if max_l > greatest:
			greatest = max_l
	return greatest

print(longestSubString(stringL))
