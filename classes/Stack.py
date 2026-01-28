class Stack:
    def __init__(self, ele=None):
        self.stk = []
        self.top = -1
        
        if isinstance(ele, int):
            self.stk.append(ele)
            self.top = 0
        elif isinstance(ele, list):
            self.stk.extend(ele)
            self.top = len(ele) - 1

    def peek(self):
        if self.isEmpty():
            return None
        return self.stk[self.top]

    def push(self, ele):
        self.stk.append(ele)
        self.top += 1

    def pop(self):
        if self.isEmpty():
            raise IndexError("pop from empty stack")
        self.stk.pop()
        self.top -= 1

    def isEmpty(self):
        return self.top == -1

    def size(self):
        return self.top + 1

    def showStack(self):
        print(self.stk)
