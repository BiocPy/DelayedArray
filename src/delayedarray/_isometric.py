from typing import Literal, Tuple
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


translate_ufunc_to_op_with_args = set(
    [
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
)


translate_ufunc_to_op_simple = set(
    [
        "log",
        "log1p",
        "log2",
        "log10",
        "exp",
        "expm1",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "ceil",
        "floor",
        "trunc",
        "sign",
    ]
)


def _infer_along_with_args(shape: Tuple[int, ...], value):
    along = None
    if not isinstance(value, numpy.ndarray) or len(value.shape) == 0:
        return along

    ndim = len(shape)
    if len(value.shape) == 1:
        along = ndim - 1
        return along

    if len(value.shape) != ndim:
        raise ValueError("length of 'value.shape' and 'seed.shape' should be equal")

    for i in range(ndim):
        if value.shape[i] != 1:
            if along is not None:
                raise ValueError("no more than one entry of 'value.shape' should be greater than 1")
            if shape[i] != value.shape[i]:
                raise ValueError("any entry of 'value.shape' that is not 1 should be equal to the corresponding entry of 'seed.shape'") 
            along = i

    return along
