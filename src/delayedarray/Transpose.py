from typing import Tuple, Optional

from numpy import dtype, transpose
from .utils import _create_dask_array
from dask.array.core import Array


class Transpose:
    """Delayed transposition, based on Bioconductor's ``DelayedArray::DelayedAperm`` class.
    This will create a matrix transpose in the 2-dimensional case; for a high-dimensional array, it will permute the dimensions.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    In general, end users should not be interacting with ``Transpose`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        perm (Optional[Tuple[int, ...]]):
            Tuple of length equal to the dimensionality of ``seed``, containing the permutation of dimensions.
            If None, the dimension ordering is assumed to be reversed.
    """

    def __init__(self, seed, perm: Optional[Tuple[int, ...]]):
        self._seed = seed

        curshape = seed.shape
        ndim = len(curshape)
        if perm is not None:
            if len(perm) != ndim:
                raise ValueError(
                    "Dimensionality of 'seed' and 'perm' should be the same."
                )
        else:
            perm = (*range(ndim - 1, -1, -1),)

        self._perm = perm

        final_shape = []
        for x in perm:
            final_shape.append(curshape[x])

        self._shape = (*final_shape,)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Transpose`` object.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``Transpose`` object,
            (i.e., after transposition of ``seed``).
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``Transpose`` object. This will be the same as the ``seed``.

        Returns:
            dtype: NumPy type for the ``Transpose`` contents.
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
    def perm(self) -> Tuple[int, ...]:
        """Get the permutation of dimensions used in the transposition.

        Returns:
            Tuple[int, ...]: Permutation of dimensions.
        """
        return self._perm

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed transposition.

        Returns:
            Array: dask array with the delayed transposition.
        """
        target = _create_dask_array(self._seed)
        return transpose(target, axes=self._perm)
