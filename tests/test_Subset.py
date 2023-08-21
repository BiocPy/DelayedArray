import copy

import delayedarray
import pytest
from utils import *
import numpy
import random


def test_Subset_consecutive_dense():
    test_shape = (20, 30)
    y = numpy.random.rand(*test_shape)

    subset = (range(20), range(30))
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y).all()

    # Sliced:
    slices = ([1,3,5,7,9], [2,4,6,8])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_sorted_unique_dense():
    test_shape = (10, 20, 30)
    y = numpy.random.rand(*test_shape)

    subset = (range(2, 7), range(0, 20, 2), range(5, 25, 3))
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = ([0,2,4], [2,4,6,8], [0, 1, 2, 3, 5 ])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_sorted_dense():
    test_shape = (50, 10)
    y = numpy.random.rand(*test_shape)

    subset = ([5,5,5,10,20,20,25,30,30,30,40,45,45,45], [0,0,0,2,5,5,5,6,8,8,8,8,8,9,9])
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = ([0,2,4,6,8,10], [3,6,9,12])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_unique_dense():
    test_shape = (20, 10, 30)
    y = numpy.random.rand(*test_shape)

    subset = []
    for i in range(len(test_shape)):
        current = list(range(i, test_shape[i], i + 1))
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = (range(3, len(subset[0]), 2), range(0, len(subset[1]), 2), range(1, len(subset[2]), 2))
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_general_dense():
    test_shape = (40, 30)
    y = numpy.random.rand(*test_shape)

    subset = []
    for i in range(len(test_shape)):
        current = []
        for x in range(i, test_shape[i], 5):
            current += [x] * int(random.uniform(0, 1) * 5)
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = (range(0, len(subset[0]), 2), range(3, len(subset[1]), 2))
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_consecutive_sparse():
    test_shape = (21, 17, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (range(21), range(17), range(30))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_sparse_array(sub, (slice(None), slice(None), slice(None)))
    assert are_SparseNdarrays_equal(full, y)

    # Sliced:
    slices = ([1,3,5,7,9], [2,4,6,8], [10,12,14,16])
    partial = delayedarray.extract_sparse_array(sub, slices)
    ref = delayedarray.extract_sparse_array(y, slices)
    assert are_SparseNdarrays_equal(partial, ref)


def test_Subset_sorted_unique_sparse():
    test_shape = (50, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (range(10, 40, 3), range(1, 28, 2))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_sparse_array(sub, (slice(None), slice(None)))
    ref = delayedarray.extract_sparse_array(y, subset)
    assert are_SparseNdarrays_equal(full, ref)

    # Sliced:
    slices = ([0,2,4,6,8], [1,3,7,9,11,13])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = []
    for i in range(2):
        current = []
        for s in slices[i]:
            current.append(subset[i][s])
        refsub.append(current)
    ref = delayedarray.extract_sparse_array(y, (*refsub,))
    assert are_SparseNdarrays_equal(partial, ref)


def test_Subset_sorted_sparse():
    test_shape = (30, 20)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = ([3,6,6,9,12,12,12,15,21,24,24,24,24], [5, 6, 7, 7, 8, 14, 14, 15, 16,16,18])
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    ref = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == ref).all()

    # Sliced:
    slices = ([2,5,8,11], [2,3,4,7,8,9])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = ref[numpy.ix_(*slices)]
    assert (delayedarray.extract_dense_array(partial, full_indices) == refsub).all()


def test_Subset_unique_sparse():
    test_shape = (20, 15, 25)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = []
    for i in range(len(test_shape)):
        current = list(range(i, test_shape[i], len(test_shape) - i))
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    ref = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == ref).all()

    # Sliced:
    slices = ([2,4,6], [1,3,5], [4,5,10,15,20,21])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = ref[numpy.ix_(*slices)]
    assert (delayedarray.extract_dense_array(partial, full_indices) == refsub).all()


def test_Subset_general_sparse():
    test_shape = (25, 55)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = []
    for i in range(len(test_shape)):
        current = []
        for x in range(i, test_shape[i], 5):
            current += [x] * int(random.uniform(0, 1) * 5)
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    refsub = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == refsub).all()

    # Sliced:
    slices = (range(2, len(subset[0]), 2), range(1, len(subset[1]), 2))
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (delayedarray.extract_dense_array(partial, full_indices) == refsub[numpy.ix_(*slices)]).all()
