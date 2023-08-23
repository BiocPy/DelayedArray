import operator
import warnings
from typing import Literal, Tuple, Union

import numpy
from numpy import inf, ndarray

from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .utils import sanitize_indices, sanitize_single_index

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

OP = Literal["+", "-", "/", "*", "//", "%", "**"]


def _choose_operator(op: OP, inplace: bool = False):
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
    """Unary isometric operation involving an n-dimensional seed array with a scalar or
    1-dimensional vector.

    This is based on Bioconductor's ``DelayedArray::DelayedUnaryIsoOpWithArgs`` class.
    Only one n-dimensional array is involved here, hence the "unary" in the name.

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. We suggest supplying a floating-point ``value`` to avoid unexpected results from
    integer truncation or overflow.

    Attributes:
        seed:
            An array-like object.

        value (Union[float, ndarray]):
            A scalar or 1-dimensional array with which to perform an operation on the ``seed``.

        op (OP):
            String specifying the operation.
            This should be one of "+", "-", "/", "*", "//", "%" or "**".

        right (bool, optional):
            Whether ``value`` is to the right of ``seed`` in the operation.
            If False, ``value`` is put to the left of ``seed``.
            Ignored for commutative operations in ``op``.

        along (int, optional):
            Dimension along which the ``value`` is to be added, if ``value`` is a
            -dimensional array. This assumes that ``value`` is of length equal to the dimension's
            extent. Ignored if ``value`` is a scalar.
    """

    def __init__(
        self,
        seed,
        value: Union[float, ndarray],
        op: OP,
        right: bool = True,
        along: int = 0,
    ):
        f = _choose_operator(op)

        dummy = numpy.zeros(0, dtype=seed.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            if isinstance(value, ndarray):
                dummy = f(dummy, value[:0])
            else:
                dummy = f(dummy, value)
        dtype = dummy.dtype
        same_type = dtype == seed.dtype

        is_no_op = None
        if same_type:
            if op == "+":
                is_no_op = 0
            elif op == "*":
                is_no_op == 1
            else:
                if right:
                    if op == "-":
                        is_no_op = 0
                    elif op == "/" or op == "**":
                        is_no_op = 1

        def check(s, v):
            try:
                with warnings.catch_warnings():  # silence warnings from divide by zero.
                    warnings.simplefilter("ignore")
                    if right:
                        return f(s, v)
                    else:
                        return f(v, s)
            except ZeroDivisionError:
                return inf

        is_sparse = False
        no_op = False

        if isinstance(value, ndarray):
            if along < 0 or along >= len(seed.shape):
                raise ValueError(
                    "'along' should be non-negative and less than the dimensionality of 'seed'"
                )
            if len(value) != seed.shape[along]:
                raise ValueError(
                    "length of array 'value' should equal the 'along' dimension extent in 'seed'"
                )

            is_sparse = True
            for x in value:
                if check(0, x) != 0:
                    is_sparse = False
                    break

            no_op = True
            for x in value:
                if x != is_no_op:
                    no_op = False
                    break
        else:
            is_sparse = check(0, value) == 0
            no_op = value == is_no_op

        inplaceable = False
        if right and same_type:
            inplaceable = True

        self._seed = seed
        self._value = value
        self._op = op
        self._right = right
        self._along = along
        self._preserves_sparse = is_sparse
        self._is_no_op = no_op
        self._do_inplace = inplaceable
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        return self._dtype


@is_sparse.register
def _is_sparse_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs) -> bool:
    return x._preserves_sparse and is_sparse(x._seed)


@extract_dense_array.register
def _extract_dense_array_UnaryIsometricOpWithArgs(
    x: UnaryIsometricOpWithArgs, idx
) -> ndarray:
    base = extract_dense_array(x._seed, idx)
    if x._is_no_op:
        return base

    opfun = _choose_operator(x._op, inplace=x._do_inplace)
    if x._right:

        def f(s, v):
            return opfun(s, v)

    else:

        def f(s, v):
            return opfun(v, s)

    value = x._value
    if isinstance(value, ndarray):
        curslice = idx[x._along]
        new_idx = sanitize_single_index((curslice,), (x.shape[x._along],))[0]
        value = value[new_idx]

        # My brain too smooth to figure out how to get numpy to do this
        # quickly for me, so we'll just loop over the targeted dimension.
        if x._along < len(base.shape) - 1 and len(base.shape) > 1:
            base = base.astype(x._dtype, copy=False)
            contents = [slice(None)] * len(base.shape)
            for i in range(len(value)):
                contents[x._along] = i
                if x._do_inplace:
                    f(
                        base[(..., *contents)], value[i]
                    )  # this is a view, so inplace is fine.
                else:
                    base[(..., *contents)] = f(base[(..., *contents)], value[i])
            return base

    return f(base, value)


def _recursive_apply_op_with_arg_to_sparse_array(contents, at, ndim, op):
    if len(at) == ndim - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                contents[i] = (idx, op(idx, val, (*at, i)))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                _recursive_apply_op_with_arg_to_sparse_array(
                    contents[i], (*at, i), ndim, op
                )


@extract_sparse_array.register
def _extract_sparse_array_UnaryIsometricOpWithArgs(
    x: UnaryIsometricOpWithArgs, idx
) -> SparseNdarray:
    sparse = extract_sparse_array(x._seed, idx)
    if x._is_no_op:
        return sparse

    idx = sanitize_indices(idx, x.shape)
    opfun = _choose_operator(x._op, inplace=x._do_inplace)
    if x._right:

        def f(s, v):
            return opfun(s, v)

    else:

        def f(s, v):
            return opfun(v, s)

    other = x._value
    if isinstance(other, ndarray):
        curslice = idx[x._along]
        other = other[curslice]

        def execute(indices, values, at):
            if x._along == len(at):
                operand = other[indices]
            else:
                operand = other[at[x._along]]
            return f(values, operand)

    else:

        def execute(indices, values, at):
            return f(values, other)

    if isinstance(sparse._contents, list):
        _recursive_apply_op_with_arg_to_sparse_array(
            sparse._contents, (), len(sparse.shape), execute
        )
    elif sparse._contents is not None:
        idx, val = sparse._contents
        sparse._contents = (idx, execute(idx, val, ()))

    sparse._dtype = x._dtype
    return sparse
