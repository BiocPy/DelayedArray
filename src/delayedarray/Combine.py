from typing import Tuple, Sequence, TYPE_CHECKING
import warnings

from numpy import concatenate, dtype, ndarray

if TYPE_CHECKING:
    import dask.array

from .utils import create_dask_array, extract_array, _densify, chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Combine:
    """Delayed combine operation, based on Bioconductor's ``DelayedArray::DelayedAbind`` class.

    This will combine multiple arrays along a specified dimension, provided the extents of all other dimensions are
    the same.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end users should not be interacting with ``Combine`` objects directly.

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

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        extracted = []
        for x in self._seeds:
            extracted.append(create_dask_array(x))
        return concatenate((*extracted,), axis=self._along)

    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        # Figuring out which slices belong to who.
        chosen = subset[self._along]
        limit = 0
        fragmented = []
        position = 0
        for x in self._seeds:
            start = limit
            limit += x.shape[self._along]
            current = []
            while position < len(chosen) and chosen[position] < limit:
                current.append(chosen[position] - start)
                position += 1
            fragmented.append(current)

        extracted = []
        flexargs = list(subset)
        for i, x in enumerate(self._seeds):
            if len(fragmented[i]):
                flexargs[self._along] = fragmented[i]
                extracted.append(extract_array(x, (*flexargs,)))

        expected_shape = []
        for i, s in enumerate(subset):
            expected_shape.append(len(s))

        try:
            output = concatenate((*extracted,), axis=self.along)
            if output.shape != (*expected_shape,):
                raise ValueError(
                    "'numpy.concatenate' on "
                    + str(type(extracted[0]))
                    + " objects does not return the correct shape"
                )
        except Exception as e:
            warnings.warn(str(e))
            for i, x in enumerate(extracted):
                extracted[i] = _densify(x)
            output = concatenate((*extracted,), axis=self.along)

        return output

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        chunks = [chunk_shape(x) for x in self._seeds]

        # Not bothering with doing anything too fancy here.  We just use the
        # maximum chunk size (which might also expand, e.g., if you're
        # combining column-major and row-major matrices; oh well).  Just accept
        # that we'll probably need to break chunks during iteration.
        output = []
        for i in range(len(self._shape)):
            dim = []
            for ch in chunks:
                dim.append(ch[i])
            output.append(max(*dim))

        return (*output,) 
