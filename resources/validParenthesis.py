class Solution:
    def isValid(self, s: str) -> bool:
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}

        for char in s:
            if char in mapping:
                # Pop the top element if the stack is not empty, else assign a dummy value
                top_element = stack.pop() if stack else '#'

                # The mapping for the opening bracket in our hash and the top element of the stack don't match, return False
                if mapping[char] != top_element:
                    return False
            else:
                # We have an opening bracket, simply push it onto the stack.
                stack.append(char)

        # In the end, if the stack is empty, then we have a valid expression.
        return not stack

if __name__ == "__main__":
    solution = Solution()
    
    # Test cases
    test_cases = [
        "()",
        "()[]{}",
        "(]",
        "([)]",
        "{[]}",
        "print('Hello, World'))" # From previous context, though strictly LeetCode usually only has brackets
    ]
    
    for s in test_cases:
        print(f"Input: {s}, Valid: {solution.isValid(s)}")
