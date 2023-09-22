from typing import Sequence, Tuple, TYPE_CHECKING

from numpy import dtype, ndarray, ix_

if TYPE_CHECKING:
    import dask.array

from .utils import create_dask_array, extract_array, chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def _sanitize(subset):
    okay = True
    for i in range(1, len(subset)):
        if subset[i] <= subset[i - 1]:
            okay = False
            break

    if okay:
        return subset, None

    sortvec = []
    for i, d in enumerate(subset):
        sortvec.append((d, i))
    sortvec.sort()

    san = []
    remap = [None] * len(sortvec)
    last = None
    for d, i in sortvec:
        if last != d:
            san.append(d)
            last = d
        remap[i] = len(san) - 1

    return san, remap


class Subset:
    """Delayed subset operation, based on Bioconductor's ``DelayedArray::DelayedSubset`` class.
    This will slice the array along one or more dimensions, equivalent to the outer product of subset indices.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end users should not be interacting with ``Subset`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        subset (Tuple[Sequence[int], ...]):
            Tuple of length equal to the dimensionality of ``seed``, containing the subsetted
            elements for each dimension.
            Each entry should be a vector of integer indices specifying the elements of the
            corresponding dimension to retain, where each integer is non-negative and less than the
            extent of the dimension. Unsorted and/or duplicate indices are allowed.
    """

    def __init__(self, seed, subset: Tuple[Sequence[int], ...]):
        self._seed = seed
        if len(subset) != len(seed.shape):
            raise ValueError(
                "Dimensionality of 'seed' and 'subset' should be the same."
            )

        self._subset = subset
        final_shape = []
        for idx in subset:
            final_shape.append(len(idx))
        self._shape = (*final_shape,)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``Subset`` object. This should be the same length as the ``seed``.

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

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    @property
    def subset(self) -> Tuple[Sequence[int], ...]:
        """Get the subset of elements to extract from each dimension of the seed.

        Returns:
            Tuple[Sequence[int], ...]: Subset vectors to be applied to each dimension of the seed.
        """
        return self._subset

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        target = create_dask_array(self._seed)

        # Oh god, this is horrible. But dask doesn't support ix_ yet.
        ndim = len(target.shape)
        for i in range(ndim):
            replacement = self._subset[i]
            if isinstance(replacement, range):
                replacement = list(replacement)

            current = [slice(None)] * ndim
            current[i] = replacement
            target = target[(..., *current)]

        return target

    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        newsub = list(subset)
        expanded = []
        is_safe = 0

        for i, s in enumerate(newsub):
            cursub = self._subset[i]
            if isinstance(cursub, ndarray):
                replacement = cursub[s]
            else:
                replacement = [cursub[j] for j in s]

            san_sub, san_remap = _sanitize(replacement)
            newsub[i] = san_sub

            if san_remap is None:
                is_safe += 1
                san_remap = range(len(san_sub))
            expanded.append(san_remap)

        raw = extract_array(self._seed, (*newsub,))
        if is_safe != len(subset):
            raw = raw[ix_(*expanded)]
        return raw

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        chunk = chunk_shape(self._seed)
        full = self._shape

        # We don't bother doing anything too fancy here because the subset
        # might render the concept of rectangular chunks invalid (e.g., if the
        # subset involves reordering or duplication). We'll just cap the chunk
        # size to the matrix dimension and call it a day.  We also set lower
        # bound of 1 to ensure that iteration is always positive.
        output = []
        for i in range(len(full)):
            output.append(max(1, min(chunk[i], full[i])))

        return (*output,)

        

