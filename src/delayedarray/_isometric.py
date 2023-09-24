from typing import Literal
import numpy


ISOMETRIC_OP_WITH_ARGS = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "remainder",
    "floor_divide",
    "power",
    "equal",
    "greater_equal",
    "greater",
    "less_equal",
    "less",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
]


def _choose_operator(operation):
    # Can't use match/case yet, as that's only in Python 3.10, and we can't
    # just dispatch to 'getattr(numpy, operation)', because some classes don't
    # implement __array_func__. Thanks a lot, scipy.sparse, and fuck you.
    if operation == "add":
        return lambda left, right : left + right
    elif operation == "subtract":
        return lambda left, right : left - right
    elif operation == "multiply":
        return lambda left, right : left * right
    elif operation == "divide":
        return lambda left, right : left / right
    elif operation == "remainder":
        return lambda left, right : left % right
    elif operation == "floor_divide":
        return lambda left, right : left // right
    elif operation == "power":
        return lambda left, right : left**right
    elif operation == "equal":
        return lambda left, right : left == right
    elif operation == "greater_equal":
        return lambda left, right : left >= right
    elif operation == "greater":
        return lambda left, right : left > right
    elif operation == "less_equal":
        return lambda left, right : left <= right
    elif operation == "less":
        return lambda left, right : left < right
    elif operation == "not_equal":
        return lambda left, right : left != right
    return getattr(numpy, operation)


def _execute(left, right, operation):
    return _choose_operator(operation)(left, right)
