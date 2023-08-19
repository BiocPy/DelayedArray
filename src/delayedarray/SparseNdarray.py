from bisect import bisect_left
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy

from .utils import sanitize_indices

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class SparseNdarray:
    """The ``SparseNdarray``, as its name suggests, is a sparse n-dimensional array.
    It is inspired by the **SVTArray** from the DelayedArray
    `R/Bioconductor package <https://github.com/Bioconductor/DelayedArray>`_.

    Internally, the ``SparseNdarray`` is represented as a nested list where each
    nesting level corresponds to a dimension. The list at each level has length equal
    to the extent of the dimension, where each entry contains another list representing
    the contents of the corresponding element of that dimension. This proceeds until
    the penultimate dimension, where each entry contains ``(index, value)`` tuples
    representing the sparse contents of the corresponding dimension element.

    In effect, this is a tree where the non-leaf nodes are lists and the leaf nodes
    are tuples. ``index`` and ``value`` should be :py:class:`~typing.Sequence` of equal
    length, where ``index`` is integer, sorted, and contains values less than the extent
    of the final dimension.

    Any entry of any list may also be None, indicating that the corresponding element
    of the dimension contains no non-zero values. In fact, the entire tree may be None,
    indicating that there are no non-zero values in the entire array.

    For `1-dimensional` arrays, the array is represented by a single (index, value) tuple
    containing the sparse contents. This may also be None if there are no non-zero
    values in the array.

    Attributes:
        shape (Tuple[int, ...]):
            Tuple containing the dimensions of the array.

        contents (Union[Tuple[Sequence, Sequence], List], optional):
            For `n-dimensional` arrays where n > 1, a nested list representing a
            tree where each leaf node is a tuple containing a sparse vector (or None).

            For `1-dimensional` arrays, a tuple containing a sparse vector.

            Alternatively None, if the array is empty.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        contents: Optional[
            Union[
                Tuple[Sequence, Sequence],
                List,
            ]
        ],
        check=True,
    ):
        self._shape = shape
        self._contents = contents

        if check is True and contents is not None:
            if len(shape) > 1:
                _recursive_check(self._contents, 0, self._shape)
            else:
                _check_sparse_tuple(
                    self._contents[0], self._contents[1], self._shape[0]
                )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array.

        Returns:
            Tuple[int, ...]: Tuple of integers containing the array shape along
            each dimension.
        """
        return self._shape

    def __get_item__(self, args: Tuple[Union[slice, Sequence], ...]) -> "SparseNdarray":
        """Extract sparse array by slicing this data array.

        Args:
            args (Tuple[Union[slice, Sequence], ...]): A :py:class`tuple` defining the
                positions of the array to access along each dimension.

                Each element in ``args`` may be a :py:func:`slice` object or
                a list of integer indices. The length of the tuple
                must not exceed the number of dimensions in the array.

        Raises:
            ValueError: If ``args`` contain more dimensions than the shape of the array.

        Returns:
            SparseNdarray: A SparseNdarray of the given indices.
        """
        if len(args) > len(self.shape):
            raise ValueError("Slice exceeds the available dimensions in the array.")

        return _extract_sparse_array_from_SparseNdarray(self, args)


def _check_sparse_tuple(indices: Sequence, values: Sequence, max_index: int):
    if len(indices) != len(values):
        raise ValueError("Length of index and value vectors should be the same.")

    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= max_index:
            raise ValueError("Index vectors out of range for the last dimension.")

    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            raise ValueError("Index vectors should be sorted.")


def _recursive_check(contents: Sequence, dim: int, shape: Tuple[int, ...]):
    if len(contents) != shape[dim]:
        raise ValueError(
            "Length of 'contents' or its components should match the extent of the corresponding dimension."
        )

    ndim = len(shape)
    if dim == ndim - 2:
        for x in contents:
            if x is not None:
                _check_sparse_tuple(x[0], x[1], shape[ndim - 1])
    else:
        for x in contents:
            if x is not None:
                _recursive_check(x, dim + 1, shape)


def _characterize_indices(idx: Sequence):
    if len(idx) == 0:
        return (idx, None, None, False)

    first = idx[0]
    last = idx[-1] + 1
    consecutive = True
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            consecutive = False
            break
    return (idx, first, last, consecutive)


def _extract_sparse_vector_internal(
    indices: Sequence,
    values: Sequence,
    idx_summary: Tuple[Sequence, int, int, int],
    f: Callable,
    last_dim_shape: int,
):
    idx, first, last, consecutive = idx_summary
    if len(idx) == 0:
        return

    start_pos = 0
    if first:
        start_pos = bisect_left(indices, first)

    if consecutive:
        end_pos = len(indices)
        if last != last_dim_shape:
            end_pos = bisect_left(indices, last, lo=start_pos, hi=end_pos)
        for x in range(start_pos, end_pos):
            f(indices[x] - first, indices[x], values[x])
    else:
        pos = 0
        x = start_pos
        xlen = len(indices)
        for i in idx:
            while x < xlen and i > indices[x]:
                x += 1
            if x == xlen:
                break
            if i == indices[x]:
                f(pos, i, values[x])
            pos += 1


def _extract_sparse_vector_to_dense(
    indices, values, idx_summary, output, last_dim_shape
):
    def f(p, i, v):
        output[p] = v

    _extract_sparse_vector_internal(indices, values, idx_summary, f, last_dim_shape)


def _recursive_extract_dense_array(contents, ndim, idx, dim, output, last_dim_shape):
    curdex = idx[dim]

    if dim == ndim - 2:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _extract_sparse_vector_to_dense(
                    x[0], x[1], idx[ndim - 1], output[pos], last_dim_shape
                )
            pos += 1
    else:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _recursive_extract_dense_array(
                    x, ndim, idx, dim + 1, output[pos], last_dim_shape
                )
            pos += 1


def _extract_dense_array_from_SparseNdarray(
    x: SparseNdarray, idx: Tuple[Union[slice, Sequence], ...]
) -> numpy.ndarray:
    idx2 = sanitize_indices(idx, x.shape)
    idims = [len(y) for y in idx2]
    idx2[-1] = _characterize_indices(idx2[-1])

    output = numpy.zeros((*idims,))
    ndims = len(x.shape)
    if ndims > 1:
        _recursive_extract_dense_array(x._contents, ndims, idx2, 0, output, x.shape[-1])
    else:
        _extract_sparse_vector_to_dense(
            x._contents[0], x._contents[1], idx2[0], output, x.shape[-1]
        )
    return output


def _extract_sparse_vector_to_sparse(indices, values, idx_summary, last_dim_shape):
    new_indices = []
    new_values = []

    def f(p, i, v):
        new_indices.append(p)
        new_values.append(v)

    _extract_sparse_vector_internal(indices, values, idx_summary, f, last_dim_shape)

    if len(new_indices) == 0:
        return None

    if isinstance(indices, numpy.ndarray):
        new_indices = numpy.array(new_indices, dtype=indices.dtype)
    if isinstance(values, numpy.ndarray):
        new_values = numpy.array(new_values, dtype=values.dtype)
    return new_indices, new_values


def _recursive_extract_sparse_array(contents, shape, idx, dim, last_dim_shape):
    ndim = len(shape)
    curdex = idx[dim]
    new_contents = []

    if dim == ndim - 2:
        pos = 0
        last_idx = idx[ndim - 1]
        for i in curdex:
            x = contents[i]
            if x is not None:
                y = _extract_sparse_vector_to_sparse(
                    x[0], x[1], last_idx, last_dim_shape
                )
                new_contents.append(y)
            else:
                new_contents.append(None)
            pos += 1
    else:
        pos = 0
        for i in curdex:
            if contents[i] is not None:
                y = _recursive_extract_sparse_array(
                    contents[i], shape, idx, dim + 1, last_dim_shape
                )
                new_contents.append(y)
            else:
                new_contents.append(None)
            pos += 1

    for x in new_contents:
        if x is not None:
            return new_contents
    return None


def _extract_sparse_array_from_SparseNdarray(
    x: SparseNdarray, idx: Tuple[Union[slice, Sequence], ...]
) -> SparseNdarray:
    idx2 = sanitize_indices(idx, x.shape)
    idims = [len(y) for y in idx2]
    idx2[-1] = _characterize_indices(idx2[-1])

    new_contents = None
    if x._contents is not None:
        if len(x.shape) > 1:
            new_contents = _recursive_extract_sparse_array(
                x._contents, x._shape, idx2, 0, x.shape[-1]
            )
        else:
            new_contents = _extract_sparse_vector_to_sparse(
                x._contents[0], x._contents[1], idx2[0], x.shape[-1]
            )

    return SparseNdarray(shape=(*idims,), contents=new_contents)
