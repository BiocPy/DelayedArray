from typing import Tuple, Sequence, TYPE_CHECKING

from numpy import dtype

if TYPE_CHECKING:
    import dask.array

from .DelayedOp import DelayedOp
from .utils import create_dask_array, extract_array, _retry_single, chunk_shape, is_sparse

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

    def __DelayedArray_extract_dense__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_dense_array.extract_dense_array`."""
        return extract_dense_array(self.seed, subset).astype(self._dtype, copy=False)

    def __DelayedArray_extract_sparse__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
        return extract_sparse_array(self.seed, subset).astype(self._dtype, copy=False)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self.seed)

    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return is_sparse(self._seed)

