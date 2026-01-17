
string: str = "A man, a plan, a canal: Panama"

def validatePalindrome(case: str) -> bool:
	string_n = ""
	for i in case:
		if i.isalnum():
			string_n += i.lower()
	return string_n == string_n[::-1]

print(validatePalindrome(string))
