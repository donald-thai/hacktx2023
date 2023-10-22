def increase_ord_by_x(input_string, x):
    result = ""
    for char in input_string:
        # increase if not space or new line
        new_ord = ord(char) + x if char not in [" ", "\t", "\n"] else ord(char)
        result += chr(new_ord)
    return result

# Input string
input_string = """import torch as t

class bruh(Module):
    __init__(self):
        self.bruh_module = self.super()

    print("yes")"""
x = int(input("Enter the value to increase by (e.g., 1): "))

# Call the function and print the result
result_string = increase_ord_by_x(input_string, x)
print(result_string)
