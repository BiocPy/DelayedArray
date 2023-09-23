import copy

import delayedarray
import pytest
import numpy

#######################################################
#######################################################

import random

def mock_SparseNdarray_contents(
    shape, density1=0.5, density2=0.5, lower=-1, upper=1, dtype=numpy.float64
):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.uniform(lower, upper))
        if len(new_values):
            return numpy.array(new_indices), numpy.array(new_values, dtype=dtype)
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
                        shape[1:],
                        density1=density1,
                        density2=density2,
                        lower=lower,
                        upper=upper,
                        dtype=dtype,
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


#######################################################
#######################################################


def test_SparseNdarray_check():
    test_shape = (10, 15, 20)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape
    assert y.dtype == numpy.float64

    with pytest.raises(ValueError, match="match the extent"):
        y = delayedarray.SparseNdarray((5, 15, 20), contents)

    with pytest.raises(ValueError, match="out of range"):
        y = delayedarray.SparseNdarray((10, 15, 1), contents)

    def scramble(con, depth):
        if depth == len(test_shape) - 2:
            for x in con:
                if x is not None:
                    i, v = x
                    random.shuffle(i)
        else:
            for x in con:
                if x is not None:
                    scramble(x, depth + 1)

    contents2 = copy.deepcopy(contents)
    scramble(contents2, 0)
    with pytest.raises(ValueError, match="should be sorted"):
        y = delayedarray.SparseNdarray(test_shape, contents2)

    def shorten(con, depth):
        if depth == len(test_shape) - 2:
            for i in range(len(con)):
                if con[i] is not None:
                    con[i] = (con[i][0][:-1], con[i][1])
        else:
            for x in con:
                if x is not None:
                    shorten(x, depth + 1)

    contents2 = copy.deepcopy(contents)
    shorten(contents2, 0)
    with pytest.raises(ValueError, match="should be the same"):
        y = delayedarray.SparseNdarray(test_shape, contents2)

    with pytest.raises(ValueError, match="Inconsistent data type"):
        y = delayedarray.SparseNdarray(test_shape, contents, dtype=numpy.int32)

    with pytest.raises(ValueError, match="cannot infer 'dtype'"):
        y = delayedarray.SparseNdarray(test_shape, None)

    empty = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.int32)
    assert empty.shape == test_shape
    assert empty.dtype == numpy.int32


def test_SparseNdarray_extract_dense_array_3d():
    test_shape = (16, 32, 8)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_dense_array(
        y, (slice(None), slice(None), slice(None))
    )
    assert (
        output == convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    ).all()

    # Sliced extraction.
    indices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()

    indices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()

    indices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()

    indices = (slice(10, 30), slice(None), slice(None))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()


def test_SparseNdarray_extract_dense_array_2d():
    test_shape = (50, 100)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_dense_array(y, (slice(None), slice(None)))
    assert (
        output == convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    ).all()

    # Sliced extraction.
    indices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()

    indices = (slice(20, 30), slice(None))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()

    indices = (slice(None), slice(10, 80))
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()


def test_SparseNdarray_extract_dense_array_1d():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.dtype == numpy.float64

    # Full extraction.
    output = delayedarray.extract_dense_array(y, (slice(None),))
    assert (
        output == convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    ).all()

    # Sliced extraction.
    indices = (slice(5, 90, 7),)
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()


def test_SparseNdarray_extract_sparse_array_3d():
    test_shape = (20, 15, 10)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(
        y, (slice(None), slice(None), slice(None))
    )
    assert are_SparseNdarray_contents_equal(output._contents, contents, len(test_shape))

    ref = convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    full_indices = (slice(None), slice(None), slice(None))

    # Sliced extraction.
    indices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()

    indices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()

    indices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()

    indices = (slice(10, 30), slice(None), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()


def test_SparseNdarray_extract_sparse_array_2d():
    test_shape = (99, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(y, (slice(None), slice(None)))
    assert are_SparseNdarray_contents_equal(output._contents, contents, len(test_shape))

    ref = convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    full_indices = (slice(None), slice(None))

    # Sliced extraction.
    indices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()

    indices = (slice(20, 30), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()

    indices = (slice(None), slice(10, 80))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()


def test_SparseNdarray_extract_sparse_array_1d():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(y, (slice(None),))
    assert are_SparseNdarray_contents_equal(output._contents, contents, len(test_shape))

    ref = convert_SparseNdarray_contents_to_numpy(contents, test_shape)
    full_indices = (slice(None),)

    # Sliced extraction.
    indices = (slice(5, 90, 7),)
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (
        delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]
    ).all()


def test_SparseNdarray_int_type():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(
        test_shape, lower=-100, upper=100, dtype=numpy.int16
    )
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape
    assert y.dtype == numpy.int16

    dout = delayedarray.extract_dense_array(y, (slice(None), slice(None)))
    assert dout.dtype == numpy.int16
    ref = convert_SparseNdarray_to_numpy(y)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(y, (slice(None), slice(None)))
    assert spout.dtype == numpy.int16
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()


def test_SparseNdarray_empty():
    test_shape = (20, 21, 22)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.uint32)
    assert y.shape == test_shape
    assert y.dtype == numpy.uint32

    dout = delayedarray.extract_dense_array(y, (slice(None), slice(None), slice(None)))
    assert (dout == numpy.zeros(test_shape)).all()
    dout = delayedarray.extract_dense_array(
        y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12])
    )
    assert (dout == numpy.zeros((3, 4, 5))).all()

    spout = delayedarray.extract_sparse_array(
        y, (slice(None), slice(None), slice(None))
    )
    assert spout._contents is None
    assert spout.shape == test_shape
    assert spout.dtype == numpy.uint32
    spout = delayedarray.extract_sparse_array(
        y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12])
    )
    assert spout.shape == (3, 4, 5)


def test_SparseNdarray_0d():
    y = delayedarray.SparseNdarray((), None, dtype=numpy.uint32)
    assert y.shape == ()
    assert y.dtype == numpy.uint32

    y = delayedarray.SparseNdarray((), 5, dtype=numpy.uint32)
    assert y.shape == ()
    assert y.dtype == numpy.uint32

    with pytest.raises(ValueError, match="0-dimensional"):
        y = delayedarray.SparseNdarray((), {}, dtype=numpy.uint32)
