import delayedarray
import numpy
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Transpose_simple(mask_rate):
    y = simulate_ndarray((30, 23), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    t = x.T
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert delayedarray.chunk_shape(t) == (23, 1)
    assert not delayedarray.is_sparse(t)
    assert delayedarray.is_masked(t) == (mask_rate > 0)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), y.T)

    t = numpy.transpose(x)
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert not delayedarray.is_sparse(t)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), numpy.transpose(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Transpose_more_dimensions(mask_rate):
    y = simulate_ndarray((30, 23, 10), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    t = numpy.transpose(x, axes=(1, 2, 0))
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 10, 30)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), numpy.transpose(y, axes=(1, 2, 0)))

    t = numpy.transpose(x)
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (10, 23, 30)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), numpy.transpose(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Transpose_subset(mask_rate):
    y = simulate_ndarray((30, 23, 10), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    t = numpy.transpose(x)

    subset = (range(2, 8), range(3, 16), range(4, 24))
    assert_identical_ndarrays(delayedarray.extract_dense_array(t, subset), y.T[numpy.ix_(*subset)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Transpose_sparse(mask_rate):
    y = simulate_SparseNdarray((30, 23), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)

    t = numpy.transpose(x)
    assert delayedarray.is_sparse(t)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), densed.T)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Transpose_dask(mask_rate):
    y = simulate_ndarray((30, 23, 10), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    t = numpy.transpose(x)

    import dask
    da = delayedarray.create_dask_array(t)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(t), da.compute())
