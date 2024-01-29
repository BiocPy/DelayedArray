from typing import Callable, Tuple, Sequence
import numpy
from numpy import dtype

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


class Round(DelayedOp):
    """Delayed rounding, resulting from :py:meth:`~numpy.round`. This is very similar to
    :py:class:`~UnaryIsometricOpSimple` but accepts an argument for the number of decimal places.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End users should not be interacting with ``Round`` objects directly.
    """

    def __init__(self, seed, decimals: int):
        """
        Args:
            seed:
                Any object that satisfies the seed contract,
                see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

            decimals (int):
                Number of decimal places, possibly negative.
        """
        self._seed = seed
        self._decimals = decimals

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of the
            ``Round`` object. This is the same as the ``seed`` array.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """

        Returns:
            NumPy type for the ``Round``, same as the ``seed`` array.
        """
        return self._seed.dtype

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed

    @property
    def decimals(self) -> int:
        """
        Returns:
            Number of decimal places to round to.
        """
        return self._decimals


def _extract_array(x: Round, subset: Tuple[Sequence[int], ...], f: Callable):
    target = f(x._seed, subset)
    return numpy.round(target, decimals=x._decimals)


@extract_dense_array.register
def extract_dense_array_Round(x: Round, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return _extract_array(x, subset, extract_dense_array)


@extract_sparse_array.register
def extract_sparse_array_Round(x: Round, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)


@create_dask_array.register
def create_dask_array_Round(x: Round):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = create_dask_array(x._seed)
    return numpy.round(target, decimals=x._decimals)


@chunk_shape.register
def chunk_shape_Round(x: Round):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return chunk_shape(x._seed)


@is_sparse.register
def is_sparse_Round(x: Round):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return is_sparse(x._seed)


@is_masked.register
def is_masked_Round(x: Round):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed)
