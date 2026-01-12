def reverse(x: int) -> int:
	s = str(x)
	if x < 0:
		res = -int(s[:0:-1])
	else:
		res = int(s[::-1])

	if res < -2147483648 or res > 2147483647:
		return 0
	return res


print(reverse(123))
print(reverse(-153420))
