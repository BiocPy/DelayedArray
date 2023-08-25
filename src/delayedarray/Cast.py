from typing import Tuple

from numpy import dtype
from .utils import _create_dask_array
from dask.array.core import Array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Cast:
    """Delayed cast to a different NumPy type. This is most useful for promoting integer matrices to floating point to
    avoid problems with integer overflow in arithmetic operations.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    End users should not be interacting with ``Cast`` objects directly.

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
        return self._dtype

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed cast.

        Returns:
            Array: dask array with the delayed cast.
        """
        target = _create_dask_array(self._seed)
        return target.astype(self._dtype)

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed
