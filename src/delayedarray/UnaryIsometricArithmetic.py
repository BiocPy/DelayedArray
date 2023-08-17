from collections import Sequence
from .interface import extract_dense_array, extract_sparse_array
from .SparseNdarray import SparseNdarray
import numpy

class UnaryIsometricArihmetic:
    """Unary isometric arithmetic, where a scalar or 1-dimensional array is added to a seed array.

    Attributes:
        seed: 
            An array-like object.

        value: 
            A scalar or 1-dimensional array to operate on the ``seed``.

        op (str):
            String specifying the operation.
            This should be one of "+", "-", "/", "*", "//", "%" or "**".

        right (bool, optional):
            Whether ``value`` is to the right of ``seed`` in the operation.
            If False, ``value`` is put to the left of ``seed``.

        along (int, optional): 
            Dimension along which the ``value`` is to be added, if ``value`` is a 1-dimensional array.
    """
    def __init__(self, seed, value, op, right: bool = True, along: int = 0):
        is_sparse = False 
        no_op = False 

        if isinstance(value, collections.Sequence):
            if len(value) != seed.shape[along]:
                raise ValueError("length of sequence-like 'value' should equal seed dimension in 'dim'")

        if right:
            if isinstance(value, collections.Sequence):
                if op == "+" or op == "-":
                    is_sparse = True
                    no_op = True
                    for x in value:
                        if x:
                            is_sparse = False
                            no_op = False
                            break

                elif op == "*" or op == "/" or op == "%" or op == "**":
                    no_op = True
                    for x in value:
                        if x != 1:
                            no_op = False
                            break

                    if op == "*":
                        is_sparse = True
                        for x in value:
                            if not numpy.isfinite(x):
                                is_sparse = False
                                break

                    elif op == "/" or op == "%":
                        is_sparse = True
                        for x in value:
                            if x == 0:
                                is_sparse = False
                                break

                    elif op == "**":
                        is_sparse = True
                        for x in value:
                            if not numpy.isfinite(x) and x > 0:
                                is_sparse = False
                                break

            else:
                if op == "+" or op == "-":
                    if value == 0:
                        is_sparse = True
                        no_op = True
                        
                elif op == "*":
                    is_sparse = numpy.isfinite(value)
                    no_op = (value == 1)

                elif op == "/" or op == "%":
                    is_sparse = op != 0
                    no_op = (value == 1)

                elif op == "**":
                    is_sparse = x > 0 and numpy.isfinite(value)
                    no_op = (value == 1)


        self._seed = seed
        self._value = value
        self._op = op
        self._right = right
        self._along = along
        self._sparse = is_sparse
        self._no_op = no_op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._seed.shape

    @property
    def sparse(self) -> bool:
        if self._seed.sparse:
            return self._sparse
        else:
            return False

def _apply_arith_to_numpy(x, value, op):
    if op == "+":
        x += value
    elif op == "-":
        x -= value
    elif op == "*":
        x *= value
    elif op == "/":
        x /= value
    elif op == "//":
        x //= value
    elif op == "%":
        x %= value
    elif op == "**":
        x **= value
    else:
        raise ValueError("unknown operation '" + op +"'")

@extract_dense_array.UnaryIsometricArithmetic
def extract_dense_array_UnaryIsometricArithmetic(x: UnaryIsometricArithemtic, idx) -> numpy.ndarray:
    base = extract_dense_array(x._seed, idx)
    if x._no_op:
        return base

    value = x._value
    if isinstance(value, collections.Sequence):
        curslice = idx[x._along]
        if curslice:
            value = value[curslice]

        if isinstance(value, numpy.ndarray) and along == len(dense.shape):
            pass
        else:
            # brain too smooth to figure out how to get numpy to do this quickly for me.
            # I also can't just use an OP() here, because the LHS could become a scalar
            # and then there's no point.
            contents = [slice(None)] * len(base.shape)
            for i in range(len(value)):
                contents[along] = i
                if op == "+":
                    base[(..., *contents)] += value[i]
                elif op == "-":
                    if x._right:
                        base[(..., *contents)] -= value[i]
                    else:
                        base[(..., *contents)] = value[i] - base[(..., *contents)]
                elif op == "*":
                    base[(..., *contents)] *= value[i]
                elif op == "/":
                    if x._right:
                        base[(..., *contents)] /= value[i]
                    else:
                        base[(..., *contents)] = value[i] / base[(..., *contents)]
                elif op == "//":
                    if x._right:
                        base[(..., *contents)] //= value[i]
                    else:
                        base[(..., *contents)] = value[i] / base[(..., *contents)]
                elif op == "%":
                    if x._right:
                        base[(..., *contents)] %= value[i]
                    else:
                        base[(..., *contents)] = value[i] % base[(..., *contents)]
                elif op == "**":
                    if x._right:
                        base[(..., *contents)] **= value[i]
                    else:
                        base[(..., *contents)] = value[i] ** base[(..., *contents)]
                else:
                    raise ValueError("unknown operation '" + op +"'")
            return base

    _apply_arith_to_numpy(base, value, op)
    return base

def _recursive_apply_op_with_arg_to_sparse_array(contents, at, ndim, op):
    if dim == ndim - 2:
        for i in range(len(contents)):
            x = contents[i]
            if x is not None:
                op(x[0], x[1], (*at, i))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                _recursive_apply_arg_to_sparse_array(contents[i], value, along, (*at, i), ndim, op)


@extract_sparse_array.UnaryIsometricArithmetic
def extract_sparse_array_UnaryIsometricArithmetic(x: UnaryIsometricArithmetic, idx) -> SparseNdarray:
    sparse = extract_sparse_array(x.__seed, idx)
    if x._no_op:
        return sparse

    value = x._value
    if isinstance(value, collections.Sequence):
        curslice = idx[x._along]
        if curslice:
            value = value[curslice]

        def execute(indices, values, at):
            if x._along == len(at):
                sub = value[indices]
                _apply_arith_to_numpy(values, value[indices], x._op)
            else:
                _apply_arith_to_numpy(values, value[at[x._along]], x._op)

    else:
        def execute(indices, values, pos):
            _apply_arith_to_numpy(values, x._value, x._op)
        _recursive_apply_op_with_arg_to_sparse_array(sparse._contents, (,), len(x.shape), execute):
