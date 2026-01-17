# Longest substring without repeating character

stringL = "pwwkew"

def longestSubString(string_test:str) -> int:
	char_map = {}
	max_length = 0
	start = 0

	for end, char in enumerate(string_test):
		if char in char_map and char_map[char] >= start:
			start = char_map[char] + 1
		char_map[char] = end
		max_length = max(max_length, end - start + 1)

	return max_length

print(longestSubString(stringL))