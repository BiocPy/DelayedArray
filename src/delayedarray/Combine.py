from typing import Callable, Tuple, Sequence
import numpy

from .DelayedOp import DelayedOp
from ._mask import _concatenate_unmasked_ndarrays, _concatenate_maybe_masked_ndarrays
from .extract_dense_array import extract_dense_array
from .SparseNdarray import _concatenate_SparseNdarrays
from .extract_sparse_array import extract_sparse_array
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class Combine(DelayedOp):
    """Delayed combine operation, based on Bioconductor's ``DelayedArray::DelayedAbind`` class.

    This will combine multiple arrays along a specified dimension, provided the extents of all other dimensions are
    the same.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end users should not be interacting with ``Combine`` objects directly.
    """

    def __init__(self, seeds: list, along: int):
        """
        Args:
            seeds:
                List of objects that satisfy the seed contract,
                see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

            along:
                Dimension along which the seeds are to be combined.
        """

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
            to_combine.append(numpy.ndarray((0,), dtype=seeds[i].dtype))
        self._dtype = _concatenate_unmasked_ndarrays(to_combine, axis=0).dtype

        self._is_masked = any(is_masked(y) for y in self._seeds)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of the
            object after seeds were combined along the specified dimension.
        """
        return self._shape

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type for the combined data.  This may or may not be
            the same as those in ``seeds``, depending on casting rules.
        """
        return self._dtype

    @property
    def seeds(self) -> list:
        """
        Returns:
            List of seed objects to be combined.
        """
        return self._seeds

    @property
    def along(self) -> int:
        """
        Returns:
            Dimension along which the seeds are combined.
        """
        return self._along


def _extract_subarrays(x: Combine, subset: Tuple[Sequence[int], ...], f: Callable):
    # Figuring out which slices belong to who.
    chosen = subset[x._along]
    limit = 0
    fragmented = []
    position = 0
    for s in x._seeds:
        start = limit
        limit += s.shape[x._along]
        current = []
        while position < len(chosen) and chosen[position] < limit:
            current.append(chosen[position] - start)
            position += 1
        fragmented.append(current)

    # Extracting the desired slice from each seed.
    extracted = []
    flexargs = list(subset)
    for i, s in enumerate(x._seeds):
        if len(fragmented[i]):
            flexargs[x._along] = fragmented[i]
            extracted.append(f(s, (*flexargs,)))

    return extracted


@extract_dense_array.register
def extract_dense_array_Combine(x: Combine, subset: Tuple[Sequence[int], ...]):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    fragments = _extract_subarrays(x, subset, extract_dense_array)
    return _concatenate_maybe_masked_ndarrays(fragments, axis=x._along, masked=x._is_masked)


@extract_sparse_array.register
def extract_sparse_array_Combine(x: Combine, subset: Tuple[Sequence[int], ...]):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    fragments = _extract_subarrays(x, subset, extract_sparse_array)
    return _concatenate_SparseNdarrays(fragments, along=x._along)


@create_dask_array.register
def create_dask_array_Combine(x: Combine):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    extracted = []
    for s in x._seeds:
        extracted.append(create_dask_array(s))
    return numpy.concatenate((*extracted,), axis=x._along)


@chunk_shape.register
def chunk_shape_Combine(x: Combine):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    chunks = [chunk_shape(s) for s in x._seeds]

    # Not bothering with doing anything too fancy here.  We just use the
    # maximum chunk size (which might also expand, e.g., if you're
    # combining column-major and row-major matrices; oh well).  Just accept
    # that we'll probably need to break chunks during iteration.
    output = []
    for i in range(len(x._shape)):
        dim = []
        for ch in chunks:
            dim.append(ch[i])
        output.append(max(*dim))

    return (*output,) 


@is_sparse.register
def is_sparse_Combine(x: Combine):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    for s in x._seeds:
        if not is_sparse(s):
            return False
    return len(x._seeds) > 0


@is_masked.register
def is_masked_Combine(x: Combine):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return x._is_masked
