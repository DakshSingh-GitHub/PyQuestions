
l1 = [9, 9, 9, 9, 9, 9, 9]
l2 = [9, 9, 9, 9]

num1_st = ""
num2_st = ""

for i in l1[::-1]: num1_st += str(i)
for i in l2[::-1]: num2_st += str(i)

res_no = str(int(num1_st) + int(num2_st))

out = []
for i in res_no[::-1]: out.append(int(i))
print(out)
