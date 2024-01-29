import numpy
import delayedarray
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Cast_simple(mask_rate):
    test_shape = (30, 23)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    assert isinstance(z.seed, delayedarray.Cast)
    assert z.dtype == numpy.dtype("int32")
    assert z.shape == test_shape
    assert delayedarray.chunk_shape(z) == (1, 23)
    assert not delayedarray.is_sparse(z)
    assert delayedarray.is_masked(z) == (mask_rate > 0)

    assert_identical_ndarrays(delayedarray.to_dense_array(z), y.astype(numpy.int32))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Cast_subset(mask_rate):
    test_shape = (30, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)

    z = x.astype(numpy.int32)
    ref = y.astype(numpy.int32)
    subset = (range(10, 20), range(5, 15))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, subset), ref[numpy.ix_(*subset)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Cast_sparse(mask_rate):
    y = simulate_SparseNdarray((10, 20), mask_rate=mask_rate, density1=0.1) * 100 # scaling it up that coercion doesn't create an all-zero matrix.
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)

    z = x.astype(numpy.int32)
    assert delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), densed.astype(numpy.int32))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Cast_dask(mask_rate):
    test_shape = (30, 23)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), da.compute())
