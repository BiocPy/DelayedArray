from collections import Sequence
from .interface import extract_dense_array, extract_sparse_array
from .SparseNdarray import SparseNdarray
import numpy
import operator

def _choose_operator(op: str, inplace: bool = False):
    if op == "+":
        if inplace:
            return operator.iadd
        else:
            return operator.add
    elif op == "-":
        if inplace:
            return operator.isub
        else:
            return operator.sub
    elif op == "*":
        if inplace:
            return operator.imul
        else:
            return operator.mul
    elif op == "/":
        if inplace:
            return operator.itruediv
        else:
            return operator.truediv
    elif op == "//":
        if inplace:
            return operator.ifloordiv
        else:
            return operator.floordiv
    elif op == "%":
        if inplace:
            return operator.imod
        else:
            return operator.mod
    elif op == "**":
        if inplace:
            return operator.ipow
        else:
            return operator.pow
    else: 
        raise ValueError("unknown operation '" + op + "'")


class UnaryIsometricOpWithArgs:
    """Unary isometric operation involving an n-dimensional seed array with a scalar or 1-dimensional vector.
    Only n-dimensional array is involved here, hence the "unary" in the name.
    Hey, I don't make the rules.

    Attributes:
        seed: 
            An array-like object.

        value (Union[float, numpy.ndarray]): 
            A scalar or 1-dimensional array with which to perform an operation on the ``seed``.

        op (str):
            String specifying the operation.
            This should be one of "+", "-", "/", "*", "//", "%" or "**".

        right (bool, optional):
            Whether ``value`` is to the right of ``seed`` in the operation.
            If False, ``value`` is put to the left of ``seed``.

        along (int, optional): 
            Dimension along which the ``value`` is to be added, if ``value`` is a 1-dimensional array.
    """
    def __init__(self, seed, value: Union[float, numpy.ndarray], op: str, right: bool = True, along: int = 0):
        is_sparse = False 
        no_op = False 

        is_no_op = None
        if op == "+" or op == "-":
            is_no_op = 0
        elif op == "*" or op == "/" or op == "%" or op == "**":
            is_no_op = 1

        f = _choose_operator(op)
        def check(s, v):
            if right:
                return f(s, v)
            else:
                return f(v, s)

        if isinstance(value, collections.Sequence):
            if len(value) != seed.shape[along]:
                raise ValueError("length of sequence-like 'value' should equal seed dimension in 'dim'")

                is_sparse = True
                for x in value:
                    if check(0, x):
                        is_sparse = False
                        break

                no_op = True
                for x in value:
                    if x != is_no_op:
                        no_op = False
                        break
        else:
            is_sparse = check(0, value)
            no_op = value == is_no_op

        inplaceable = False
        if x._right: # TODO: add something about types.
            inplaceable = True

        self._seed = seed
        self._value = value
        self._op = op
        self._right = right
        self._along = along
        self._preserves_sparse = is_sparse
        self._is_no_op = no_op
        self._do_inplace = inplaceable

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._seed.shape

    @property
    def sparse(self) -> bool:
        if self._seed.sparse:
            return self._preserves_sparse
        else:
            return False


@extract_dense_array.UnaryIsometricArithmetic
def extract_dense_array_UnaryIsometricArithmetic(x: UnaryIsometricArithemtic, idx) -> numpy.ndarray:
    base = extract_dense_array(x._seed, idx)
    if x._is_no_op:
        return base

    opfun = _choose_operator(op, inplace = x._do_inplace)
    if x._right:
        def f(s, v):
            return opfun(s, v)
    else:
        def f(s, v):
            return opfun(v, s)

    value = x._value
    if isinstance(value, numpy.ndarray):
        curslice = idx[x._along]
        if curslice:
            value = value[curslice]

        if along < len(base.shape) and len(base.shape) > 1:
            # My brain too smooth to figure out how to get numpy to do this
            # quickly for me. I also can't just use an OP() here, because
            # the LHS could become a scalar and then there's no point.
            contents = [slice(None)] * len(base.shape)
            for i in range(len(value)):
                contents[along] = i
                if x._do_inplace:
                    f(base[(..., *contents)], value[i]) # this is a view, so inplace is fine.
                else:
                    base[(..., *contents)] = f(base[(..., *contents)], value[i]) 
            return base

    return f(base, value)


def _recursive_apply_op_with_arg_to_sparse_array(contents, at, ndim, op):
    if dim == ndim - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                contents[i] = (idx, op(idx, val, (*at, i)))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                _recursive_apply_arg_to_sparse_array(contents[i], (*at, i), ndim, op)


@extract_sparse_array.UnaryIsometricArithmetic
def extract_sparse_array_UnaryIsometricArithmetic(x: UnaryIsometricArithmetic, idx) -> SparseNdarray:
    sparse = extract_sparse_array(x.__seed, idx)
    if x._is_no_op:
        return sparse

    opfun = _choose_operator(op, inplace = x._do_inplace)
    if x._right:
        def f(s, v):
            return opfun(s, v)
    else:
        def f(s, v):
            return opfun(v, s)

    other = x._value
    if isinstance(other, collections.Sequence):
        curslice = idx[x._along]
        if curslice:
            other = other[curslice]

        def execute(indices, values, at):
            if x._along == len(at):
                operand = other[indices]
            else:
                operand = other[x._along]]
            return f(values, operand)
    else:
        def execute(indices, values, at):
            return f(values, other)

    if isinstance(sparse._contents, list):
        _recursive_apply_op_with_arg_to_sparse_array(sparse._contents, (,), len(sparse.shape), execute):
    elif sparse._contents is not None:
        idx, val = sparse._contents
        sparse._contents = (idx, execute(idx, val, (,)))
    return sparse
