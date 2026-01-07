
string = "Daksh is a good boy"

def reverseString(string_r:str):
	spst = string_r.split(" ")
	res = []
	for i in spst:
		res.append(i[::-1])
	return " ".join(res)

print(reverseString(string))
