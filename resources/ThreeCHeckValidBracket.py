def checkParenthesis(string: str) -> bool:
	stack = []
	mapping = {')': '(', ']': '[', '}': '{'}

	for char in string:
		if char in mapping.values():
			stack.append(char)
		elif char in mapping:
			if not stack or stack.pop() != mapping[char]:
				return False

	return len(stack) == 0

if __name__ == "__main__":
	sentence = "Hello() This[] is{ a string"
	condition = checkParenthesis(sentence)
	print(condition)
