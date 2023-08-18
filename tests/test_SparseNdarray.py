import copy

import delayedarray
import pytest
from utils import *


def test_SparseNdarray_check():
    test_shape = (10, 15, 20)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape

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
