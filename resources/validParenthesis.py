# Stack method

# from classes.Stack import Stack
# def isValid(s: str) -> bool:
#     stack = Stack()
#     mapping = {")": "(", "}": "{", "]": "["}
#     for char in s:
#         if char in mapping.values():
#             stack.push(char)
#         elif char in mapping.keys():
#             if stack.isEmpty() or mapping[char] != stack.peek():
#                 return False
#             stack.pop()
#         else:
#             continue
#     return stack.isEmpty()
#
# if __name__ == "__main__":
#     string = "print('Hello, World'))"
#     if isValid(string):
#         print("Balanced")
#     else:
#         print("Unbalanced")


# Balance factor method
string = "()"

balance_factor = 0
balanced = True

for i in range(len(string)):
	if string[i] == "(":
		balance_factor += 1
	elif string[i] == ")":
		balance_factor -= 1
		if balance_factor < 0:
			balanced = False
			break

if balanced and balance_factor == 0:
	print("Balanced")
else:
	print("Unbalanced")

