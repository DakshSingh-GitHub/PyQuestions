
ROMAN_MAP = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
}

def romanToInteger(roman: str) -> int:
    total = 0
    prev_value = 0
    for char in reversed(roman):
        value = ROMAN_MAP[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
        
    return total
