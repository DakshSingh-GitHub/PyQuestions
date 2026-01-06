
string1 = "listen"
string2 = "silent"

def anagramChecker(st1, st2):
	if sorted(st1) == sorted(st2):
		return True
	return False

print(anagramChecker(string1, string2))
