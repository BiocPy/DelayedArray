from typing import Tuple

import numpy
from numpy import dtype
from .utils import _create_dask_array
from dask.array.core import Array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Round:
    """Delayed rounding, resulting from :py:meth:`~numpy.round`. This is very similar to
    :py:class:`~UnaryIsometricOpSimple` but accepts an argument for the number of decimal places.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    End users should not be interacting with ``Round`` objects directly.

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

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed rounding operation.

        Returns:
            Array: dask array with the delayed rounding operation.
        """
        target = _create_dask_array(self._seed)
        return numpy.round(target, decimals=self._decimals)

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
