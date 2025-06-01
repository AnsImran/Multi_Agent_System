from typing import Literal

def arithmetic_calculator(n1: int | float, n2: int | float, ops: Literal["add", "subtract", "multiply", "divide"]) -> int | float:
    """
    Takes in two numbers as input and performs an arithmetic operation on them.
    args:
            n1:  1st int or float
            n2:  2nd int or float
            ops: the arithmetic operation to perform. supported operations are "add", "subtract", "multiply" and "divide" 
    """
    if ops.lower() == 'add':
        return n1+n2
    elif ops.lower() == 'subtract':
        return n1-n2
    elif ops.lower() == 'multiply':
        return n1*n2
    elif ops.lower() == 'divide':
        return n1/n2
    else:
        raise ValueError(f"Invalid operation: {ops}")

