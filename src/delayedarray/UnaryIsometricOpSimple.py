from typing import Callable, Literal, Tuple, Sequence
import numpy
from numpy import dtype, zeros

from .DelayedOp import DelayedOp
from .SparseNdarray import SparseNdarray
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

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
    "logical_not",
]


def _choose_operator(op: OP):
    return getattr(numpy, op)


class UnaryIsometricOpSimple(DelayedOp):
    """Delayed unary isometric operation involving an n-dimensional seed array with no additional arguments,
    similar to Bioconductor's ``DelayedArray::DelayedUnaryIsoOpStack`` class.
    This is used for simple mathematical operations like NumPy's :py:meth:`~numpy.log`.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End-users should not be interacting with ``UnaryIsometricOpSimple`` objects directly.
    """

    def __init__(self, seed, operation: OP):
        """
        Args:
            seed:
                Any object that satisfies the seed contract,
                see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

            operation:
                String specifying the unary operation.
        """
        f = _choose_operator(operation)
        dummy = f(zeros(1, dtype=seed.dtype))

        self._seed = seed
        self._op = operation
        self._dtype = dummy.dtype
        self._sparse = is_sparse(self._seed) and dummy[0] == 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of the
            object after the operation. This should be the same as ``seed``.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type for the contents of the object after the operation.
            This may or may not be the same as the ``seed`` array, depending on
            how NumPy does the casting for the requested operation.
        """
        return self._dtype

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed

    @property
    def operation(self) -> OP:
        """
        Returns:
            Name of the operation.
        """
        return self._op


def _extract_array(x: UnaryIsometricOpSimple, subset: Tuple[Sequence[int], ...], f: Callable):
    target = f(x._seed, subset)
    g = _choose_operator(x._op)
    return g(target)


@extract_dense_array.register
def extract_dense_array_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return _extract_array(x, subset, extract_dense_array)


@extract_sparse_array.register
def extract_sparse_array_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)


@create_dask_array.register
def create_dask_array_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = create_dask_array(x._seed)
    f = _choose_operator(x._op)
    return f(target)


@chunk_shape.register
def chunk_shape_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return chunk_shape(x._seed)


@is_sparse.register
def is_sparse_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return x._sparse


@is_masked.register
def is_masked_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed)
