string: str = "swiss"

def firstNonRepeatingCharacter(sequence: str) -> str:
	char_count = {}
	for char in sequence:
		char_count[char] = char_count.get(char, 0) + 1
	for char in sequence:
		if char_count[char] == 1:
			return char
	return "None"

print(firstNonRepeatingCharacter(string))