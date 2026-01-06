
stringTest = "hello, i am daksh singh"

def frequencyCounter(string:str):
	out = {}
	spc = "/*-+.!@#$%^&*(){}_+:~`<>:[]=-"
	for letter in string:
		if letter in out.keys() and (letter not in spc or not letter.isspace()):
			out[letter] += 1
		else: out[letter] = 1
	return out

print(frequencyCounter(stringTest))
