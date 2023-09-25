from typing import Callable, Optional, Tuple, Sequence, TYPE_CHECKING
from numpy import dtype
if TYPE_CHECKING:
    import dask.array

from .DelayedOp import DelayedOp
from .utils import create_dask_array, chunk_shape, is_sparse
from .extract_dense_array import extract_dense_array, _sanitize_to_fortran
from .extract_sparse_array import extract_sparse_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Cast(DelayedOp):
    """Delayed cast to a different NumPy type. This is most useful for promoting integer matrices to floating point to
    avoid problems with integer overflow in arithmetic operations.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End users should not be interacting with ``Cast`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        dtype (dtype):
            The desired type.
    """

    def __init__(self, seed, dtype: dtype):
        self._seed = seed
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Cast`` object. This is the same as the ``seed`` object.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``Cast`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``Cast`` object.

        Returns:
            dtype: NumPy type for the ``Cast`` contents.
        """
        return dtype(self._dtype)

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        target = create_dask_array(self._seed)
        return target.astype(self._dtype)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self.seed)

    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return is_sparse(self._seed)


def _extract_array(x: Cast, subset: Optional[Tuple[Sequence[int]]], f: Callable)
    return f(x._seed, subset).astype(x._dtype, copy=False)


@extract_dense_array.register
def extract_dense_array_Cast(x: Cast, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.utils.extract_dense_array.extract_dense_array`."""
    out = _extract_array(x, subset, extract_dense_array)
    return _sanitize_to_fortran(out)


@extract_sparse_array.register
def extract_sparse_array_Cast(x: Cast, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)
