
d1c = {"a": 1, "b": 2, "c": 3}
d2c = {"c": 5, "d": 4, "e": 5}

def dictionaryAddition(d1:dict, d2:dict):
	out = {}
	for i in d1: out[i] = [d1[i]]
	for i in d2:
		if i in out:
			out[i].append(d2[i])
		else:
			out[i] = [d2[i]]
	return out


print(dictionaryAddition(d1c, d2c))
