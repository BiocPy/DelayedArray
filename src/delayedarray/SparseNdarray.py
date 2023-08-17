from typing import Sequence, Tuple
import numpy

class SparseNdarray:
    def __init__(self, shape, contents, check = True):
        self._shape = shape
        self._contents = contents

        if check and contents is not None:
            if len(shape) > 1:
                _recursive_check(self._contents, 0, self._shape)
            else:
                _check_sparse_tuple(self._contents[0], self._contents[1], self._shape[0])

    @property
    def sparse(self):
        return True

    @property
    def shape(self):
        return self._shape 


def _check_sparse_tuple(indices, values, max_index):
    if len(indices) != len(values):
        raise ValueError("length of index and value vectors should be the same")

    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= max_index:
            raise ValueError("index vectors out of range for the last dimension")

    for i in range(1, len(indices)):
        if indices[i] <= indices[i-1]:
            raise ValueError("index vectors should be sorted")


def _recursive_check(contents, dim, shape):
    if len(contents) != shape[dim]:
        raise ValueError("length of 'contents' or its components should match the extent of the corresponding dimension")

    ndim = len(shape)
    if dim == ndim - 2:
        for x in contents:
            if x is not None:
                _check_sparse_tuple(x[0], x[1], shape[ndim - 1])
    else:
        for x in contents:
            if x is not None:
                _recursive_check(x, dim + 1, shape)


def _extract_sparse_vector_to_dense(indices, values, idx, output):
    pos = 0
    x = 0
    xlen = len(indices)
    for i in idx:
        while x < xlen and i > indices[x]:
            x += 1
        if x == xlen:
            break
        if i == indices[x]:
            output[pos] = values[x]
        pos += 1


def _recursive_extract_dense_array(contents, ndim, idx, dim, output):
    curdex = idx[dim]

    if dim == ndim - 2:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _extract_sparse_vector_to_dense(x[0], x[1], idx[ndim - 1], output[pos])
            pos += 1
    else:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _recursive_extract_dense_array(x, ndim, idx, dim + 1, output[pos])
            pos += 1


def _extract_dense_array_from_SparseNdarray(x: SparseNdarray, idx: Tuple[Sequence, ...]) -> numpy.ndarray:
    idx2 = []
    for i in range(len(idx)):
        curidx = idx[i]
        if isinstance(curidx, slice):
            idx2.append(range(*curidx.indices(x.shape[i])))
        else:
            idx2.append(curidx)

    idims = [len(y) for y in idx2]
    output = numpy.zeros((*idims,))
    ndims = len(x.shape)
    if ndims > 1:
        _recursive_extract_dense_array(x._contents, ndims, idx2, 0, output)
    else:
        _extract_sparse_vector_to_dense(x._contents[0], x._contents[1], idx2[0], output)
    return output


def _extract_sparse_vector_to_sparse(indices, values, idx, output):
    pos = 0
    x = 0
    xlen = len(indices)
    new_indices = []
    new_values = []

    for i in idx:
        while x < xlen and i > indices[x]:
            x += 1
        if x == xlen:
            break
        if i == indices[x]:
            new_indices.push(pos)
            new_values.push(values[x])
        pos += 1

    if len(new_indices) == 0:
        return None

    if isinstance(indices, numpy.ndarray):
        new_indices = numpy.array(new_indices, dtype=indices.dtype)
    if isinstance(values, numpy.ndarray):
        new_values = numpy.array(new_values, dtype=values.dtype)
    return new_indices, new_values

def _recursive_extract_sparse_array(contents, idx, dim):
    ndim = len(shape)
    curdex = idx[dim]
    new_contents = []

    if dim == ndim - 2:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                y = _extract_sparse_dense_vector(x[0], x[1], idx[ndim - 1])
                new_contents.append(y)
            pos += 1
    else:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                y = _recursive_extract_dense_array(x, idx, dim + 1)
                new_contents.append(y)
            pos += 1

    for x in new_contents:
        if x is not None:
            return new_contents
    return None

def _extract_sparse_array_from_SparseNdarray(x: SparseNdarray, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    new_contents = None
    if x._contents is not None:
        if len(x.shape) > 1:
            new_contents = _recursive_extract_sparse_array(x._contents, idx, 0)
        else:
            new_contents = _extract_sparse_vector_to_sparse(x._contents[0], x._contents[1], idx[0])

    idims = [len(y) for y in idx]
    return SparseNdarray(shape = (*idims,), contents = new_contents)

