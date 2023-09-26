from typing import Optional, Callable, Literal, Tuple, Sequence, TYPE_CHECKING
import numpy
from numpy import dtype, zeros
if TYPE_CHECKING:
    import dask.array

from .DelayedOp import DelayedOp
from .utils import create_dask_array, chunk_shape, is_sparse
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array

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


class UnaryIsometricOpSimple(DelayedOp):
    """Delayed unary isometric operation involving an n-dimensional seed array with no additional arguments,
    similar to Bioconductor's ``DelayedArray::DelayedUnaryIsoOpStack`` class.
    This is used for simple mathematical operations like NumPy's :py:meth:`~numpy.log`.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End-users should not be interacting with ``UnaryIsometricOpSimple`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        operation (str):
            String specifying the unary operation.
    """

    def __init__(self, seed, operation: OP):
        f = _choose_operator(operation)
        dummy = f(zeros(1, dtype=seed.dtype))

        self._seed = seed
        self._op = operation
        self._dtype = dummy.dtype
        self._sparse = is_sparse(self._seed) and dummy[0] == 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``UnaryIsometricOpSimple`` object. As the name of the class suggests, this is the same as the
        ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``UnaryIsometricOpSimple``
            object.
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

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    @property
    def operation(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Name of the operation.
        """
        return self._op

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        target = create_dask_array(self._seed)
        f = _choose_operator(self._op)
        return f(target)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self._seed)

    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return self._sparse


def _extract_array(x: UnaryIsometricOpSimple, subset: Optional[Tuple[Sequence[int]]], f: Callable):
    target = f(self._seed, subset)
    g = _choose_operator(self._op)
    return g(target)


@extract_dense_array.register
def extract_dense_array_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.utils.extract_dense_array.extract_dense_array`."""
    out = _extract_array(x, subset, extract_dense_array)
    return _sanitize_to_fortran(out)


@extract_sparse_array.register
def extract_sparse_array_UnaryIsometricOpSimple(x: UnaryIsometricOpSimple, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)
