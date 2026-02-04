# We have to find the second-largest element without sorting

# arr = [1, 2, 4, 5, 9, 6, 0]

# largest = 0
# second_largest = 0

# for i in range(len(arr)):
#     if arr[i] > largest:
#         second_largest = largest
#         largest = arr[i]
#     elif arr[i] > second_largest and arr[i] != largest:
#         second_largest = arr[i]

# print("Second Largest: " + str(second_largest))

# def normalise_string(string: str) -> str:
#     return "".join([i for i in string.lower() if i.isalpha()])

# def checkHeaviness(string: str) -> str:
#     string = normalise_string(string)
#     dct_char = {"Vowel": 0, "Consonant": 0}
#     for i in string:
#         if i in "aeiou":
#             dct_char["Vowel"] += 1
#         else:
#             dct_char["Consonant"] += 1
#     if dct_char["Vowel"] > dct_char["Consonant"]:
#         return "VOWEL HEAVY"
#     else:
#         return "CONSONANT HEAVY"

# string = str(input("Enter a string: "))
# print(checkHeaviness(string))


# string = "LLRLRR"

# balance = 0
# max_balance = 0

# for i in string:
#     if i == "L":
#         balance += 1
#     else:
#         balance -= 1
#     if balance > max_balance:
#         max_balance = balance

# if balance != 0:
#     print(-1)
# else:
#     print(max_balance)


arr = [1, 1, 1]
out_arr = []

dict = {}
for i in arr:
	if i in dict:
		dict[i] += 1
	else:
		dict[i] = 1

for i in dict.keys():
	if dict[i] == 1:
		out_arr.append(i)

print(out_arr)