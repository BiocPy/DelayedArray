from typing import Literal, Tuple

import numpy
from numpy import ndarray

from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .utils import sanitize_indices

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

OP = Literal[
    "log",
    "log1p",
    "log2",
    "log10",
    "exp",
    "expm1",
    "sqrt",
    "abs",
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


def _choose_operator(op: OP):
    return getattr(numpy, op)


class UnaryIsometricOpSimple:
    """Unary isometric operation involving an n-dimensional seed array with no
    additional arguments.

    Attributes:
        seed: An array-like object.

        op (OP):
            String specifying the unary operation.
    """

    def __init__(self, seed, op: OP):
        f = _choose_operator(op)
        dummy = f(numpy.zeros(1, dtype=seed.dtype))

        self._seed = seed
        self._op = op
        self._preserves_sparse = dummy[0] == 0
        self._dtype = dummy.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        return self._dtype


@is_sparse.register
def _is_sparse_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple) -> bool:
    return x._preserves_sparse and is_sparse(x._seed)


@extract_dense_array.register
def _extract_dense_array_UnaryIsometricOpSimple(
    x: UnaryIsometricOpSimple, idx
) -> ndarray:
    idx = sanitize_indices(idx, x.shape)
    base = extract_dense_array(x._seed, idx)
    opfun = _choose_operator(x._op)
    return opfun(base).astype(x._dtype, copy=False)


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
def _extract_sparse_array_UnaryIsometricOpSimple(
    x: UnaryIsometricOpSimple, idx
) -> SparseNdarray:
    idx = sanitize_indices(idx, x.shape)
    sparse = extract_sparse_array(x._seed, idx)

    opfun = _choose_operator(x._op)

    def execute(indices, values, at):
        return opfun(values)

    if isinstance(sparse._contents, list):
        _recursive_apply_op_with_arg_to_sparse_array(
            sparse._contents, (), len(sparse.shape), execute
        )
    elif sparse._contents is not None:
        idx, val = sparse._contents
        sparse._contents = (idx, execute(idx, val, ()))

    sparse._dtype = x._dtype
    return sparse
