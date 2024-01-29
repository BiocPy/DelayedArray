import numpy
import delayedarray
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray, safe_concatenate


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_simple(left_mask_rate, right_mask_rate):
    y1 = simulate_ndarray((30, 23), mask_rate=left_mask_rate)
    y2 = simulate_ndarray((50, 23), mask_rate=right_mask_rate)
    x1 = delayedarray.DelayedArray(y1)
    x2 = delayedarray.DelayedArray(y2)
    x = numpy.concatenate((x1, x2))

    assert isinstance(x.seed, delayedarray.Combine)
    assert x.shape == (80, 23)
    assert x.dtype == numpy.float64
    assert x.seed.along == 0
    assert delayedarray.chunk_shape(x) == (1, 23)
    assert not delayedarray.is_sparse(x)
    assert delayedarray.is_masked(x) == (left_mask_rate + right_mask_rate > 0)

    assert_identical_ndarrays(delayedarray.to_dense_array(x), safe_concatenate((y1, y2)))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_otherdim(left_mask_rate, right_mask_rate):
    y1 = simulate_ndarray((19, 43), dtype=numpy.dtype("int32"), mask_rate=left_mask_rate)
    y2 = simulate_ndarray((19, 57), dtype=numpy.dtype("int32"), mask_rate=right_mask_rate)
    x1 = delayedarray.DelayedArray(y1)
    x2 = delayedarray.DelayedArray(y2)
    x = numpy.concatenate((x1, x2), axis=1)

    assert isinstance(x.seed, delayedarray.Combine)
    assert x.shape == (19, 100)
    assert x.dtype == numpy.int32
    assert x.seed.along == 1
    assert delayedarray.chunk_shape(x) == (1, 57)

    assert_identical_ndarrays(delayedarray.to_dense_array(x), safe_concatenate((y1, y2), axis=1))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_subset(left_mask_rate, right_mask_rate):
    y1 = simulate_ndarray((30, 23), mask_rate=left_mask_rate)
    y2 = simulate_ndarray((50, 23), mask_rate=right_mask_rate)
    x1 = delayedarray.DelayedArray(y1)
    x2 = delayedarray.DelayedArray(y2)

    z = numpy.concatenate((x1, x2))
    ref = safe_concatenate((y1, y2))
    subset = (range(5, 70, 2), range(3, 20))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, subset), ref[numpy.ix_(*subset)])

    y2b = simulate_ndarray((30, 19), mask_rate=right_mask_rate)
    x2b = delayedarray.DelayedArray(y2b)
    z = numpy.concatenate((x1, x2b), axis=1)
    ref = safe_concatenate((y1, y2b), axis=1)
    subset = (range(5, 28), range(10, 40))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, subset), ref[numpy.ix_(*subset)])


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_mixed_chunks(left_mask_rate, right_mask_rate):
    y1 = simulate_ndarray((30, 23), mask_rate=left_mask_rate)
    y2 = simulate_ndarray((23, 50), dtype=numpy.dtype("int32"), mask_rate=right_mask_rate)
    x1 = delayedarray.DelayedArray(y1)
    x2 = delayedarray.DelayedArray(y2)
    x = numpy.concatenate((x1, x2.T))

    assert x.dtype == numpy.float64
    assert delayedarray.chunk_shape(x) == (50, 23)
    assert_identical_ndarrays(delayedarray.to_dense_array(x), safe_concatenate((y1, y2.T)))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_mixed_sparse(left_mask_rate, right_mask_rate):
    y1 = simulate_SparseNdarray((100, 10), density1=0.2, mask_rate=left_mask_rate)
    y2a = simulate_SparseNdarray((100, 20), density2=0.2, mask_rate=right_mask_rate)
    densed1 = delayedarray.to_dense_array(y1)
    densed2 = delayedarray.to_dense_array(y2a)

    x1 = delayedarray.DelayedArray(y1)
    x2a = delayedarray.DelayedArray(y2a)
    x = numpy.concatenate((x1, x2a), axis=1)
    assert delayedarray.is_sparse(x)
    assert_identical_ndarrays(delayedarray.to_dense_array(x), safe_concatenate((densed1, densed2), axis=1))

    y2b = simulate_ndarray((100, 20), mask_rate=right_mask_rate)
    x2b = delayedarray.DelayedArray(y2b)
    x = numpy.concatenate((x1, x2b), axis=1)
    assert not delayedarray.is_sparse(x)
    assert_identical_ndarrays(delayedarray.to_dense_array(x), safe_concatenate((densed1, y2b), axis=1))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_Combine_dask(left_mask_rate, right_mask_rate):
    y1 = simulate_ndarray((30, 23), mask_rate=left_mask_rate)
    y2 = simulate_ndarray((50, 23), mask_rate=right_mask_rate)
    x1 = delayedarray.DelayedArray(y1)
    x2 = delayedarray.DelayedArray(y2)
    x = numpy.concatenate((x1, x2))

    import dask
    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(x), da.compute())
