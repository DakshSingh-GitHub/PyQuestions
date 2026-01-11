from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummy_head = ListNode(0)
    current = dummy_head
    carry = 0

    while l1 or l2 or carry:
        # Get values from the lists, default to 0 if list is exhausted
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        # Calculate sum and new carry
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        
        # Move pointers forward
        current = current.next
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
            
    return dummy_head.next

if __name__ == "__main__":
    # Helper to create a linked list from a list of values
    def create_linked_list(arr):
        dummy = ListNode(0)
        current = dummy
        for val in arr:
            current.next = ListNode(val)
            current = current.next
        return dummy.next

    # Test case: 342 + 465 = 807
    # Input: l1 = [2,4,3], l2 = [5,6,4]
    l1 = create_linked_list([2, 4, 3])
    l2 = create_linked_list([5, 6, 4])
    
    result = addTwoNumbers(l1, l2)
    
    # Print result
    output = []
    while result:
        output.append(str(result.val))
        result = result.next
    print(" -> ".join(output))
