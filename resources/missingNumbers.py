
# Find a single Missing Number
numbers = [0, 3, 4, 5, 2, 1]
last_n = sorted(numbers)[len(numbers)-1]
miss_n =  int(last_n*(last_n+1)/2 - sum(numbers))
if miss_n == 0: miss_n = last_n + 1;
print("Missing Number:", miss_n)

# Multiple Missing Numbers
# More generalized method, previous method is specific to only when one number is missing
numbers = [0, 3, 1, 6, 4, 2, 5, 7, 9, 8]
out = []
last_n = sorted(numbers)[len(numbers)-1]
for i in range(last_n+1):
	if i not in numbers:
		out.append(i)
if not out: out.append(last_n + 1)
print("Missing Number:", out)
