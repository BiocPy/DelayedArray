from bisect import bisect_left
from collections import namedtuple
from copy import deepcopy
from typing import Sequence, Tuple

from numpy import array, dtype, ix_, ndarray
from .utils import _create_dask_array
from dask.array.core import Array


class Subset:
    """Delayed subset operation. This will slice the array along one or more dimensions, equivalent to the outer product
    of subset indices. 

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    In general, end-users should not be interacting with ``Subset`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        subset (Tuple[Sequence, ...]):
            Tuple of length equal to the dimensionality of ``seed``, containing the subsetted
            elements for each dimension.
            Each entry should be a vector of integer indices specifying the elements of the
            corresponding dimension to retain, where each integer is non-negative and less than the
            extent of the dimension. Unsorted and/or duplicate indices are allowed.
    """

    def __init__(self, seed, subset: Tuple[Sequence, ...]):
        self._seed = seed
        if len(subset) != len(seed.shape):
            raise ValueError(
                "Dimensionality of 'seed' and 'subset' should be the same."
            )

        self._subset = []
        final_shape = []
        for i in range(len(seed.shape)):
            idx = subset[i]
            if isinstance(idx, slice):
                idx = range(*idx.indices(seed.shape[i]))
            if not isinstance(idx, list):
                idx = list(idx)
            self._subset.append(idx)
            final_shape.append(len(idx))

        self._shape = (*final_shape,)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Subset`` object. This may not be the same length as the ``seed`` if dimensions were discarded.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``Subset`` object,
            (i.e., after subsetting was applied to the ``seed``).
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``Subset`` object. This will be the same as the ``seed``.

        Returns:
            dtype: NumPy type for the ``Subset`` contents.
        """
        return self._seed.dtype

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed subset.

        Returns:
            Array: dask array with the delayed subset.
        """
        target = _create_dask_array(self._seed)

        # Oh god, this is horrible. But dask doesn't support ix_ yet.
        ndim = len(target.shape)
        for i in range(ndim):
            current = [slice(None)] * ndim
            current[i] = self._subset[i]
            target = target[(..., *current)]
        return target
