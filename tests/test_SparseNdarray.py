import random
import delayedarray
import pytest
import copy
import numpy

def mock_SparseNdarray_contents(shape, density1 = 0.5, density2 = 0.5):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.gauss(0, 1))
        if len(new_values):
            return new_indices, new_values
        else:
            return None
    else:
        new_content = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density1:
                new_content.append(None)
            else:
                new_content.append(mock_SparseNdarray_contents(shape[1:], density1=density1, density2=density2))
        return new_content


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
            for x in con:
                if x is not None:
                    i, v = x
                    i.pop()
        else:
            for x in con:
                if x is not None:
                    shorten(x, depth + 1)

    contents2 = copy.deepcopy(contents)
    shorten(contents2, 0)
    with pytest.raises(ValueError, match="should be the same"):
        y = delayedarray.SparseNdarray(test_shape, contents2)


def recursive_compute_reference(contents, at, max_depth, triplets):
    if len(at) == max_depth - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                for j in range(len(idx)):
                    triplets.append(((*at, i, idx[j]), val[j]))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                recursive_compute_reference(contents[i], (*at, i), max_depth, triplets)


def compute_reference(contents, shape):
    triplets = []

    if len(shape) == 1:
        idx, val = contents
        for j in range(len(idx)):
            triplets.append(((idx[j],), val[j]))
    else:
        recursive_compute_reference(contents, (), len(shape), triplets)

    output = numpy.zeros(shape)
    for pos, val in triplets:
        output[(..., *pos)] = val
    return output


def test_SparseNdarray_extract_dense_array_3d():
    test_shape = (16, 32, 8)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_dense_array(y, (slice(None), slice(None), slice(None)))
    assert (output == compute_reference(contents, test_shape)).all()

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
    assert (output == compute_reference(contents, test_shape)).all()

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
    assert (output == compute_reference(contents, test_shape)).all()

    # Sliced extraction.
    indices = (slice(5, 90, 7),)
    sliced = delayedarray.extract_dense_array(y, indices)
    assert (sliced == output[(..., *indices)]).all()


def test_SparseNdarray_extract_sparse_array_3d():
    test_shape = (20, 15, 10)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(y, (slice(None), slice(None), slice(None)))
    assert output._contents == contents

    ref = compute_reference(contents, test_shape)
    full_indices = (slice(None), slice(None), slice(None))

    # Sliced extraction.
    indices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()

    indices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()

    indices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()

    indices = (slice(10, 30), slice(None), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()


def test_SparseNdarray_extract_sparse_array_2d():
    test_shape = (99, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(y, (slice(None), slice(None)))
    assert (output._contents == contents)

    ref = compute_reference(contents, test_shape)
    full_indices = (slice(None), slice(None))

    # Sliced extraction.
    indices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()

    indices = (slice(20, 30), slice(None))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()

    indices = (slice(None), slice(10, 80))
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()


def test_SparseNdarray_extract_sparse_array_1d():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = delayedarray.extract_sparse_array(y, (slice(None),))
    assert output._contents == contents

    ref = compute_reference(contents, test_shape)
    full_indices = (slice(None),)

    # Sliced extraction.
    indices = (slice(5, 90, 7),)
    sliced = delayedarray.extract_sparse_array(y, indices)
    assert (delayedarray.extract_dense_array(sliced, full_indices) == ref[(..., *indices)]).all()
