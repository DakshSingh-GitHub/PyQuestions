
# we have to find the longest palindrome substring in a given string s

def longestPalindrome(s: str) -> str:
    if not s:
        return ""

    def expand(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    res = ""
    for i in range(len(s)):
        # Odd length palindrome (centered at i)
        odd = expand(i, i)
        if len(odd) > len(res):
            res = odd
        # Even length palindrome (centered between i and i+1)
        even = expand(i, i + 1)
        if len(even) > len(res):
            res = even
            
    return res

if __name__ == "__main__":
    s = "cbbd"
    print(f"Longest palindrome in '{s}': {longestPalindrome(s)}")
