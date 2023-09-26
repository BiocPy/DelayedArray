from functools import singledispatch
from numpy import array, ndarray, ix_
from bisect import bisect_left
from typing import Any, Optional, Tuple, Sequence

from ._subset import _spawn_indices, _is_subset_noop, _is_subset_consecutive
from .SparseNdarray import SparseNdarray, _extract_sparse_array_from_SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_sparse_array(x: Any, subset: Optional[Tuple[Sequence[int]]] = None) -> SparseNdarray:
    """Extract the contents of ``x`` (or a subset thereof) into a
    :py:class:`~delayedarray.SparseNdarray.SparseNdarray`. This should only be
    used for ``x`` where :py:meth:`~delayedarray.utils.is_sparse` returns True.

    Args:
        x: 
            Any object with a ``__DelayedArray_extract_sparse__`` method that
            accepts a non-None ``subset`` and returns a ``SparseNdarray``
            corresponding to the outer product of the subsets.

        subset: 
            Tuple of length equal to the number of dimensions, each containing
            a sorted and unique sequence of integers specifying the elements of
            each dimension to extract. If None, all elements are extracted from
            all dimensions.

    Returns:
        SparseNdarray for the requested subset. This may be a view so callers
        should create a copy if they intend to modify it.
    """

    if hasattr(x, "__DelayedArray_extract_sparse__"):
        if subset is None:
            subset = _spawn_indices(x.shape)
        return x.__DelayedArray_extract__(subset)
    raise NotImplementedError("'extract_sparse_array(" + str(type(x)) + ")' has not yet been implemented") 


@extract_sparse_array.register
def extract_sparse_array_SparseNdarray(x: ndarray, subset: Optional[Tuple[Sequence[int]]] = None):
    if _is_subset_noop(x.shape, subset):
        subset = None
    if subset is None:
        return x
    else:
        return _extract_sparse_array_from_SparseNdarray(x, subset)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.
has_sparse = False
try:
    import scipy.sparse
    has_sparse = True
except:
    pass


if has_sparse:
    @extract_sparse_array.register
    def extract_sparse_array_csc_matrix(x: scipy.sparse.csc_matrix, subset: Optional[Tuple[Sequence[int]]] = None):
        if subset is None:
            subset = _spawn_indices(x.shape)

        final_shape = [len(s) for s in subset]
        new_contents = None
        rowsub = subset[0]
        colsub = subset[1]
        row_consecutive = _is_subset_consecutive(rowsub)

        if final_shape[0] != 0 or final_shape[1] != 0:
            first = rowsub[0]
            last = rowsub[-1] + 1
            new_contents = []

            for ci in subset[1]:
                start_pos = x.indptr[ci]
                end_pos = x.indptr[ci + 1]
                if first != 0:
                    start_pos = bisect_left(x.indices, first, lo=start_pos, hi=end_pos)

                if row_consecutive:
                    if last != x.shape[0]:
                        end_pos = bisect_left(x.indices, last, lo=start_pos, hi=end_pos)

                    if end_pos > start_pos:
                        tmp = x.indices[start_pos:end_pos]
                        if first:
                            tmp = tmp - first # don't use -=, this might modify the view by reference.
                        new_contents.append((tmp, x.data[start_pos:end_pos]))
                    else:
                        new_contents.append(None)

                else:
                    new_val = []
                    new_idx = []
                    pos = 0
                    for ri in range(start_pos, end_pos):
                        current = x.indices[ri]
                        while pos < len(rowsub) and current > rowsub[pos]:
                            pos += 1
                        if pos == len(rowsub):
                            break
                        if current == rowsub[pos]:
                            new_idx.append(current - first)
                            new_val.append(x.data[ri])
                            pos += 1

                    if len(new_val):
                        new_contents.append((array(new_idx, dtype=x.indices.dtype), array(new_val, dtype=x.dtype))) 
                    else:
                        new_contents.append(None)

        return SparseNdarray((*final_shape,), new_contents, dtype=x.dtype, index_dtype=x.indices.dtype)
