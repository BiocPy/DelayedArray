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
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y).all()

    # Sliced:
    slices = ([1,3,5,7,9], [2,4,6,8])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_consecutive_sparse():
    test_shape = (21, 17, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (range(21), range(17), range(30))
    sub = delayedarray.Subset(y, subset)
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_sparse_array(sub, (slice(None), slice(None), slice(None)))
    assert are_SparseNdarrays_equal(full, y)

    # Sliced:
    slices = ([1,3,5,7,9], [2,4,6,8], [10,12,14,16])
    partial = delayedarray.extract_sparse_array(sub, slices)
    ref = delayedarray.extract_sparse_array(y, slices)
    assert are_SparseNdarrays_equal(partial, ref)
