from typing import Callable, Tuple, Sequence
from numpy import dtype

from .DelayedOp import DelayedOp
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Cast(DelayedOp):
    """Delayed cast to a different NumPy type. This is most useful for promoting integer matrices to floating point to
    avoid problems with integer overflow in arithmetic operations.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End users should not be interacting with ``Cast`` objects directly.
    """

    def __init__(self, seed, dtype: dtype):
        """
        Args:
            seed:
                Any object that satisfies the seed contract,
                see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

            dtype:
                The desired type.
        """
        self._seed = seed
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of this
            object. This is the same as the ``seed`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type for the contents after casting.
        """
        return dtype(self._dtype)

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed


def _extract_array(x: Cast, subset: Tuple[Sequence[int], ...], f: Callable):
    return f(x._seed, subset).astype(x._dtype, copy=False)


@extract_dense_array.register
def extract_dense_array_Cast(x: Cast, subset: Tuple[Sequence[int], ...] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return _extract_array(x, subset, extract_dense_array)


@extract_sparse_array.register
def extract_sparse_array_Cast(x: Cast, subset: Tuple[Sequence[int], ...] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)


@create_dask_array.register
def create_dask_array_Cast(x: Cast):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = create_dask_array(x._seed)
    return target.astype(x._dtype)


@chunk_shape.register
def chunk_shape_Cast(x: Cast):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return chunk_shape(x._seed)


@is_sparse.register
def is_sparse_Cast(x: Cast):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return is_sparse(x._seed)


@is_masked.register
def is_masked_Cast(x: Cast):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed)
