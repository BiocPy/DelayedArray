import random

import numpy


def mock_SparseNdarray_contents(shape, density1=0.5, density2=0.5):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.gauss(0, 1))
        if len(new_values):
            return numpy.array(new_indices), numpy.array(new_values)
        else:
            return None
    else:
        new_content = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density1:
                new_content.append(None)
            else:
                new_content.append(
                    mock_SparseNdarray_contents(
                        shape[1:], density1=density1, density2=density2
                    )
                )
        return new_content


def _recursive_compute_reference(contents, at, max_depth, triplets):
    if len(at) == max_depth - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                for j in range(len(idx)):
                    triplets.append(((*at, i, idx[j]), val[j]))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                _recursive_compute_reference(contents[i], (*at, i), max_depth, triplets)


def convert_SparseNdarray_contents_to_numpy(contents, shape):
    triplets = []

    if len(shape) == 1:
        idx, val = contents
        for j in range(len(idx)):
            triplets.append(((idx[j],), val[j]))
    elif contents is not None:
        _recursive_compute_reference(contents, (), len(shape), triplets)

    output = numpy.zeros(shape)
    for pos, val in triplets:
        output[(..., *pos)] = val
    return output


def convert_SparseNdarray_to_numpy(x):
    return convert_SparseNdarray_contents_to_numpy(x._contents, x.shape)


def _compare_sparse_vectors(left, right):
    idx_l, val_l = left
    idx_r, val_r = right
    if len(idx_l) != len(idx_r):
        return False
    if not (idx_l == idx_r).all():
        return False
    if not (val_l == val_r).all():
        return False
    return True


def _recursive_compare_contents(left, right, at, max_depth):
    if len(left) != len(right):
        return False
    if len(at) == max_depth - 2:
        for i in range(len(left)):
            if left[i] is not None:
                if right[i] is None:
                    return False
                if not _compare_sparse_vectors(left[i], right[i]):
                    return False
    else:
        for i in range(len(left)):
            if left[i] is not None:
                if not _recursive_compare_contents(
                    left[i], right[i], (*at, i), max_depth
                ):
                    return False
    return True


def are_SparseNdarray_contents_equal(contents1, contents2, maxdim):
    if isinstance(contents1, list):
        if isinstance(contents2, list):
            return _recursive_compare_contents(contents1, contents2, (), maxdim)
        else:
            return False
    elif contents1 is None:
        if contents2 is None:
            return True
        else:
            return False
    else:
        return _compare_sparse_vectors(contents1, contents2)


def are_SparseNdarrays_equal(x, y):
    if x.shape != y.shape:
        return False
    return are_SparseNdarray_contents_equal(x._contents, y._contents, len(x.shape))
