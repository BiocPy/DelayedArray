from typing import Literal, Tuple, Sequence

import numpy
from numpy import ndarray, dtype, zeros

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
    """Delayed unary isometric operation involving an n-dimensional seed array with no additional arguments. This is
    used for simple mathematical operations like NumPy's :py:meth:`~numpy.log`.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    End-users should not be interacting with ``UnaryIsometricOpSimple`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        op (str):
            String specifying the unary operation.
    """

    def __init__(self, seed, op: OP):
        f = _choose_operator(op)
        dummy = f(zeros(1, dtype=seed.dtype))

        self._seed = seed
        self._op = op
        self._preserves_sparse = dummy[0] == 0
        self._dtype = dummy.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``UnaryIsometricOpSimple`` object. As the name of the class suggests, this is the same as the
        ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``UnaryIsometricOpSimple`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``UnaryIsometricOpSimple`` object. This may or may not be the same as the ``seed`` array,
        depending on how NumPy does the casting for the requested operation.

        Returns:
            dtype: NumPy type for the ``UnaryIsometricOpSimple`` contents.
        """
        return self._dtype


@is_sparse.register
def _is_sparse_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple) -> bool:
    return x._preserves_sparse and is_sparse(x._seed)


@extract_dense_array.register
def _extract_dense_array_UnaryIsometricOpSimple(
    x: UnaryIsometricOpSimple, idx: Tuple[Sequence, ...]
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
    x: UnaryIsometricOpSimple, idx: Tuple[Sequence, ...]
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
