import numpy
import delayedarray
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray 


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Round_default(mask_rate):
    y = simulate_ndarray((30, 23), mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    assert isinstance(z.seed, delayedarray.Round)
    assert z.dtype == numpy.float64
    assert z.shape == (30, 23)
    assert delayedarray.chunk_shape(z) == (1, 23)
    assert not delayedarray.is_sparse(z)
    assert delayedarray.is_masked(z) == (mask_rate > 0)

    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.round(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Round_subset(mask_rate):
    y = simulate_ndarray((30, 23), mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    ref = numpy.round(y)
    subset = (range(5, 20), range(3, 19, 2))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, subset), ref[numpy.ix_(*subset)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Round_sparse(mask_rate):
    y = simulate_SparseNdarray((30, 10), mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)

    z = numpy.round(x)
    assert delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.round(densed))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Round_decimals(mask_rate):
    y = simulate_ndarray((30, 23), mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x, decimals=1)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.round(y, decimals=1))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Round_dask(mask_rate):
    y = simulate_ndarray((30, 23), mask_rate=mask_rate) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), da.compute())
