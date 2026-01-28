from classes.Stack import Stack

def isValid(s: str) -> bool:
    """
    Checks if a string has balanced parentheses.
    """
    stack = Stack()
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in mapping.values():
            stack.push(char)
        elif char in mapping.keys():
            if stack.isEmpty() or mapping[char] != stack.peek():
                return False
            stack.pop()
        else:
            continue
            
    return stack.isEmpty()

if __name__ == "__main__":
    string = "print('Hello, World'))"
    if isValid(string):
        print("Balanced")
    else:
        print("Unbalanced")
