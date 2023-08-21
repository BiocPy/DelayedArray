import copy

import delayedarray
import pytest
from utils import *
import numpy


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
    refsub = full[numpy.ix_(*slices)]
    assert (full[numpy.ix_(*slices)] == refsub).all()


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
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    ref = delayedarray.extract_dense_array(y, subset)
    assert (full == ref).all()

    # Sliced:
    slices = ([0,2,4,6,8], [1,3,7,9,11,13])
    partial = delayedarray.extract_dense_array(sub, slices)
    refsub = full[numpy.ix_(*slices)]
    assert (full[numpy.ix_(*slices)] == refsub).all()

