from functools import singledispatch
import numpy
from bisect import bisect_left
from typing import Any, Tuple, Sequence, List, Union
from biocutils.package_utils import is_package_installed

from ._subset import _spawn_indices, _is_subset_noop, _is_subset_consecutive
from ._mask import _convert_to_unmasked_1darray, _convert_to_maybe_masked_1darray, _allocate_unmasked_ndarray, _allocate_maybe_masked_ndarray
from .SparseNdarray import SparseNdarray, _extract_sparse_array_from_SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_sparse_array(x: Any, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """Extract the contents of ``x`` (or a subset thereof) into a
    :py:class:`~delayedarray.SparseNdarray.SparseNdarray`. This should only be
    used for ``x`` where :py:meth:`~delayedarray.is_sparse.is_sparse` is True.

    Args:
        x: 
            Any array-like object containing sparse data.

        subset: 
            Tuple of length equal to the number of dimensions, each containing
            a sorted and unique sequence of integers specifying the elements of
            each dimension to extract.

    Returns:
        ``SparseNdarray`` for the requested subset. This may be a view so
        callers should create a copy if they intend to modify it.

        If :py:func:`~delayedarray.is_masked.is_masked` is True for ``x``, the
        ``SparseNdarray`` will contain NumPy ``MaskedArray``s internally.
    """
    raise NotImplementedError("'extract_sparse_array(" + str(type(x)) + ")' has not yet been implemented") 


@extract_sparse_array.register
def extract_sparse_array_SparseNdarray(x: SparseNdarray, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    if _is_subset_noop(x.shape, subset):
        return x
    else:
        return _extract_sparse_array_from_SparseNdarray(x, subset)


if is_package_installed("scipy"):
    import scipy.sparse as sp


    def _set_empty_contents(contents: List) -> Union[List, None]:
        for x in contents:
            if x is not None:
                return contents
        return None


    @extract_sparse_array.register
    def extract_sparse_array_csc_matrix(x: sp.csc_matrix, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
        """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
        final_shape = [len(s) for s in subset]
        new_contents = None

        if final_shape[0] != 0 and final_shape[1] != 0:
            rowsub = subset[0]
            row_consecutive = _is_subset_consecutive(rowsub)
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
                    for p in range(start_pos, end_pos):
                        ri = x.indices[p]
                        while pos < len(rowsub) and ri > rowsub[pos]:
                            pos += 1
                        if pos == len(rowsub):
                            break
                        if ri == rowsub[pos]:
                            new_idx.append(pos)
                            new_val.append(x.data[p])
                            pos += 1

                    if len(new_val):
                        new_contents.append((
                            _convert_to_unmasked_1darray(new_idx, dtype=x.indices.dtype), 
                            _convert_to_maybe_masked_1darray(new_val, dtype=x.data.dtype, masked=numpy.ma.isMaskedArray(x.data))
                        )) 
                    else:
                        new_contents.append(None)

            new_contents = _set_empty_contents(new_contents)

        return SparseNdarray((*final_shape,), new_contents, dtype=x.dtype, index_dtype=x.indices.dtype, is_masked=False, check=False)


    @extract_sparse_array.register
    def extract_sparse_array_csr_matrix(x: sp.csr_matrix, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
        """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
        final_shape = [len(s) for s in subset]
        new_contents = None

        if final_shape[0] != 0 and final_shape[1] != 0:
            colsub = subset[1]
            col_consecutive = _is_subset_consecutive(colsub)
            cfirst = colsub[0]
            clast = colsub[-1] + 1

            new_contents = []
            for i in range(len(colsub)):
                new_contents.append(([], []))

            for rpos, ri in enumerate(subset[0]):
                start_pos = x.indptr[ri]
                end_pos = x.indptr[ri + 1]
                if cfirst != 0:
                    start_pos = bisect_left(x.indices, cfirst, lo=start_pos, hi=end_pos)

                if col_consecutive:
                    if clast != x.shape[1]:
                        end_pos = bisect_left(x.indices, clast, lo=start_pos, hi=end_pos)

                    for p in range(start_pos, end_pos):
                        ci = x.indices[p]
                        idx, val = new_contents[ci - cfirst]
                        idx.append(rpos)
                        val.append(x.data[p])

                else:
                    cpos = 0
                    for p in range(start_pos, end_pos):
                        ci = x.indices[p]
                        while cpos < len(colsub) and ci > colsub[cpos]:
                            cpos += 1
                        if cpos == len(colsub):
                            break
                        if ci == colsub[cpos]:
                            idx, val = new_contents[cpos]
                            idx.append(rpos)
                            val.append(x.data[p])
                            cpos += 1

            for i in range(len(new_contents)):
                idx, val = new_contents[i]
                if len(idx):
                    new_contents[i] = (
                        _convert_to_unmasked_1darray(idx, dtype=x.indices.dtype), 
                        _convert_to_maybe_masked_1darray(val, dtype=x.data.dtype, masked=numpy.ma.isMaskedArray(x.data))
                    )
                else:
                    new_contents[i] = None

            new_contents = _set_empty_contents(new_contents)

        return SparseNdarray((*final_shape,), new_contents, dtype=x.dtype, index_dtype=x.indices.dtype, is_masked=False, check=False)


    @extract_sparse_array.register
    def extract_sparse_array_coo_matrix(x: sp.coo_matrix, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
        """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
        final_shape = [len(s) for s in subset]
        new_contents = None

        if final_shape[0] != 0 and final_shape[1] != 0:
            in_row = {}
            for i, y in enumerate(subset[0]):
                in_row[y] = i

            in_col = {}
            new_contents = []
            for i, y in enumerate(subset[1]):
                in_col[y] = i
                new_contents.append([])

            for i, v in enumerate(x.data):
                r = x.row[i]
                c = x.col[i]
                if r in in_row and c in in_col:
                    new_contents[in_col[c]].append((in_row[r], v))

            for i, con in enumerate(new_contents):
                if len(con):
                    con.sort()
                    shape = (len(con),)
                    idx = _allocate_unmasked_ndarray(shape, dtype=x.row.dtype)
                    val = _allocate_maybe_masked_ndarray(shape, dtype=x.data.dtype, masked=numpy.ma.isMaskedArray(x.data))
                    for j, y in enumerate(con):
                        idx[j] = y[0]
                        val[j] = y[1]
                    new_contents[i] = (idx, val)
                else:
                    new_contents[i] = None

            new_contents = _set_empty_contents(new_contents)

        return SparseNdarray((*final_shape,), new_contents, dtype=x.dtype, index_dtype=x.row.dtype, is_masked=False, check=False)
