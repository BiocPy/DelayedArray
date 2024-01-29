from typing import Callable, Sequence, Tuple
from numpy import dtype, ndarray, ix_
import numpy

from .DelayedOp import DelayedOp
from .SparseNdarray import SparseNdarray
from ._subset import _spawn_indices, _sanitize_subset
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Subset(DelayedOp):
    """Delayed subset operation, based on Bioconductor's ``DelayedArray::DelayedSubset`` class.
    This will slice the array along one or more dimensions, equivalent to the outer product of subset indices.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end users should not be interacting with ``Subset`` objects directly.
    """

    def __init__(self, seed, subset: Tuple[Sequence[int], ...]):
        """
        Args:
            seed:
                Any object that satisfies the seed contract, see
                :py:class:`~delayedarray.DelayedArray.DelayedArray` for
                details.

            subset:
                Tuple of length equal to the dimensionality of ``seed``,
                containing the subsetted elements for each dimension.  Each
                entry should be a vector of integer indices specifying the
                elements of the corresponding dimension to retain, where each
                integer is non-negative and less than the extent of the
                dimension. Unsorted and/or duplicate indices are allowed.
        """
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
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of the
            subsetted object.
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type for the subsetted contents, same as ``seed``.
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
    def subset(self) -> Tuple[Sequence[int], ...]:
        """
        Returns:
            Subset sequences to be applied to each dimension of the seed.
        """
        return self._subset


def _extract_array(x: Subset, subset: Tuple[Sequence[int], ...], f: Callable):
    if subset is None:
        subset = _spawn_indices(x.shape)

    newsub = list(subset)
    expanded = []
    is_safe = 0

    for i, s in enumerate(newsub):
        cursub = x._subset[i]
        if isinstance(cursub, ndarray):
            replacement = cursub[s]
        else:
            replacement = [cursub[j] for j in s]

        san_sub, san_remap = _sanitize_subset(replacement)
        newsub[i] = san_sub

        if san_remap is None:
            is_safe += 1
            san_remap = range(len(san_sub))
        expanded.append(san_remap)

    raw = f(x._seed, (*newsub,))
    if is_safe != len(subset):
        raw = raw[ix_(*expanded)]
    return raw


@extract_dense_array.register
def extract_dense_array_Subset(x: Subset, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return _extract_array(x, subset, extract_dense_array)


@extract_sparse_array.register
def extract_sparse_array_Subset(x: Subset, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)


@create_dask_array.register
def create_dask_array_Subset(x: Subset):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = create_dask_array(x._seed)

    # Oh god, this is horrible. But dask doesn't support ix_ yet.
    ndim = len(target.shape)
    for i in range(ndim):
        replacement = x._subset[i]
        if isinstance(replacement, range):
            replacement = list(replacement)

        current = [slice(None)] * ndim
        current[i] = replacement
        target = target[(..., *current)]

    return target


@chunk_shape.register
def chunk_shape_Subset(x: Subset):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    chunk = chunk_shape(x._seed)
    full = x._shape

    # We don't bother doing anything too fancy here because the subset
    # might render the concept of rectangular chunks invalid (e.g., if the
    # subset involves reordering or duplication). We'll just cap the chunk
    # size to the matrix dimension and call it a day.  We also set lower
    # bound of 1 to ensure that iteration is always positive.
    output = []
    for i in range(len(full)):
        output.append(max(1, min(chunk[i], full[i])))

    return (*output,)


@is_sparse.register
def is_sparse_Subset(x: Subset):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return is_sparse(x._seed)


@is_masked.register
def is_masked_Subset(x: Subset):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed)
