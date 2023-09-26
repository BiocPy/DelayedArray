from typing import Optional, Callable, Tuple, Sequence, TYPE_CHECKING
import numpy
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


class Round(DelayedOp):
    """Delayed rounding, resulting from :py:meth:`~numpy.round`. This is very similar to
    :py:class:`~UnaryIsometricOpSimple` but accepts an argument for the number of decimal places.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End users should not be interacting with ``Round`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        decimals (int):
            Number of decimal places, possibly negative.
    """

    def __init__(self, seed, decimals: int):
        self._seed = seed
        self._decimals = decimals

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Round`` object. This is the same as the ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``Round`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``Round`` object, same as the ``seed`` array.

        Returns:
            dtype: NumPy type for the ``Round`` contents.
        """
        return self._seed.dtype

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    @property
    def decimals(self) -> int:
        """Number of decimal places to round to.

        Returns:
            int: Number of decimal places.
        """
        return self._decimals

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        target = create_dask_array(self._seed)
        return numpy.round(target, decimals=self._decimals)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self._seed)

    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return is_sparse(self._seed)


def _extract_array(x: Round, subset: Optional[Tuple[Sequence[int]]], f: Callable):
    target = f(x._seed, subset)
    return numpy.round(target, decimals=x._decimals)


@extract_dense_array.register
def extract_dense_array_Round(x: Round, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.utils.extract_dense_array.extract_dense_array`."""
    out = _extract_array(x, subset, extract_dense_array)
    return _sanitize_to_fortran(out)


@extract_sparse_array.register
def extract_sparse_array_Round(x: Round, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)
