from typing import Tuple

from numpy import dtype, ndarray, concatenate
from .utils import _create_dask_array
from dask.array.core import Array


class Combine:
    """Delayed combine operation, based on Bioconductor's ``DelayedArray::DelayedAbind`` class.
    This will combine multiple arrays along a specified dimension, provided the extents of all other dimensions are the same.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    In general, end users should not be interacting with ``Combine`` objects directly.

    Attributes:
        seeds (list):
            List of objects that satisfy the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        along (int):
            Dimension along which the seeds are to be combined.
    """

    def __init__(self, seeds: list, along: int):
        self._seeds = seeds
        if len(seeds) == 0:
            raise ValueError("expected at least one object in 'seeds'")

        shape = list(seeds[0].shape)
        ndim = len(shape)

        for i in range(1, len(seeds)):
            curshape = seeds[i].shape
            for d in range(ndim):
                if d == along:
                    shape[d] += curshape[d]
                elif shape[d] != curshape[d]:
                    raise ValueError(
                        "expected seeds to have the same extent for non-'along' dimensions"
                    )

        self._shape = (*shape,)
        self._along = along

        # Guessing the dtype.
        to_combine = []
        for i in range(len(seeds)):
            to_combine.append(ndarray((0,), dtype=seeds[i].dtype))
        self._dtype = concatenate((*to_combine,)).dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Combine`` object.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``Combine`` object,
            (i.e., after seeds were combined along the ``along`` dimension).
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``Combine`` object. This may or may not be the same as those in ``seeds``, depending on casting
        rules.

        Returns:
            dtype: NumPy type for the ``Combine`` contents.
        """
        return self._dtype

    @property
    def seeds(self) -> list:
        """Get the list of underlying seed objects.

        Returns:
            list: List of seeds.
        """
        return self._seeds

    @property
    def along(self) -> int:
        """Dimension along which the seeds are combined.

        Returns:
            int: Dimension to combine along.
        """
        return self._along

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed combination of arrays.

        Returns:
            Array: dask array with the delayed combination.
        """
        extracted = []
        for x in self._seeds:
            extracted.append(_create_dask_array(x))
        return concatenate((*extracted,), axis=self._along)
