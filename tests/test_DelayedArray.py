import numpy
import delayedarray
import pytest

from utils import simulate_ndarray, simulate_SparseNdarray, assert_close_ndarrays, assert_identical_ndarrays


def test_DelayedArray_dense():
    raw = (numpy.random.rand(40, 30) * 5 - 10).astype(numpy.int32)
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype
    assert not delayedarray.is_sparse(x)
    assert delayedarray.chunk_shape(x) == (1, 30)

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'int32'") != -1

    dump = numpy.array(x)
    assert isinstance(dump, numpy.ndarray)
    assert (dump == raw).all()


def test_DelayedArray_dask():
    raw = (numpy.random.rand(40, 30) * 5 - 10).astype(numpy.int32)
    x = delayedarray.DelayedArray(raw)
    dump = numpy.array(x)

    import dask.array
    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert (dump == da.compute()).all()


def test_DelayedArray_colmajor():
    raw = numpy.random.rand(30, 40).T
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype
    assert delayedarray.chunk_shape(x) == (40, 1)

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'float64'") != -1


def test_DelayedArray_wrap():
    raw = numpy.random.rand(30, 40)
    x = delayedarray.wrap(raw)
    assert isinstance(x, delayedarray.DelayedArray)
    assert x.shape == raw.shape
    x = delayedarray.wrap(x)
    assert isinstance(x, delayedarray.DelayedArray)


def test_DelayedArray_sparse():
    import scipy.sparse
    y = scipy.sparse.csc_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.wrap(y)

    out = delayedarray.to_sparse_array(x)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert delayedarray.chunk_shape(x) == (3, 1)
    assert delayedarray.is_sparse(x)


def test_DelayedArray_masked():
    raw = numpy.random.rand(30, 40)
    y = numpy.ma.MaskedArray(raw, raw > 0.5)
    x = delayedarray.wrap(y)
    assert delayedarray.is_masked(x)


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_sum_dense(mask_rate, buffer_size):
    raw = simulate_ndarray((30, 40, 15), mask_rate = mask_rate)
    assert_identical_ndarrays(raw.sum(), delayedarray.wrap(raw).sum())
    assert_identical_ndarrays(raw.sum(axis=0), delayedarray.wrap(raw).sum(axis=0))

    y = delayedarray.wrap(raw) * 5
    ref = raw * 5
    assert numpy.isclose(ref.sum(), y.sum(buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=1), y.sum(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=-1), y.sum(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=(0, 2)), y.sum(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_ndarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) * 5
    ref = raw * 5
    assert numpy.isclose(ref.sum(), y.sum(buffer_size=buffer_size))

    # Full masking is respected.
    y = delayedarray.wrap(numpy.ma.MaskedArray([1], mask=True)) * 5
    assert y.sum() is numpy.ma.masked


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_sum_sparse(mask_rate, buffer_size):
    raw = simulate_SparseNdarray((20, 30, 25), mask_rate = mask_rate)
    y = delayedarray.wrap(raw) * 10
    ref = raw * 10

    assert numpy.isclose(ref.sum(), y.sum(buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=1), y.sum(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=-1), y.sum(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.sum(axis=(0, 2)), y.sum(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_SparseNdarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) * 10
    ref = raw * 10
    assert numpy.isclose(ref.sum(), y.sum(buffer_size=buffer_size))

    # Full masking is respected.
    ref = delayedarray.SparseNdarray((1,), (numpy.zeros(1, dtype=numpy.int_), numpy.ma.MaskedArray([1], mask=True)))
    y = delayedarray.wrap(ref) * 10
    assert y.sum() is numpy.ma.masked


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_mean_dense(mask_rate, buffer_size):
    raw = simulate_ndarray((30, 40, 15), mask_rate = mask_rate)
    assert_identical_ndarrays(raw.mean(), delayedarray.wrap(raw).mean())
    assert_identical_ndarrays(raw.mean(axis=0), delayedarray.wrap(raw).mean(axis=0))

    y = delayedarray.wrap(raw) - 12
    ref = raw - 12
    assert numpy.isclose(ref.mean(), y.mean(buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=1), y.mean(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=-1), y.mean(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=(0, 2)), y.mean(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_ndarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) + 29
    ref = raw + 29
    assert numpy.isclose(ref.mean(), y.mean(buffer_size=buffer_size))

    # Full masking is respected.
    y = delayedarray.wrap(numpy.ma.MaskedArray([1], mask=True)) + 20
    assert y.mean() is numpy.ma.masked

    # Zero-length array is respected.
    with pytest.warns(RuntimeWarning):
        y = delayedarray.wrap(numpy.ndarray((10, 0))) * 50
        assert numpy.isnan(y.mean())


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_mean_sparse(mask_rate, buffer_size):
    raw = simulate_SparseNdarray((20, 30, 25), mask_rate = mask_rate)
    ref = raw * 19
    y = delayedarray.wrap(raw) * 19

    assert numpy.isclose(ref.mean(), y.mean(buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=1), y.mean(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=-1), y.mean(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.mean(axis=(0, 2)), y.mean(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_SparseNdarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) * 12
    ref = raw * 12
    assert numpy.isclose(ref.mean(), y.mean(buffer_size=buffer_size))

    # Full masking is respected.
    ref = delayedarray.SparseNdarray((1,), (numpy.zeros(1, dtype=numpy.int_), numpy.ma.MaskedArray([1], mask=True)))
    y = delayedarray.wrap(ref) / 5
    assert y.mean() is numpy.ma.masked

    # Zero-length array is respected.
    with pytest.warns(RuntimeWarning):
        y = delayedarray.wrap(delayedarray.SparseNdarray((0,), None, dtype=numpy.int32, index_dtype=numpy.int32)) * 50
        assert numpy.isnan(y.mean())


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_var_dense(mask_rate, buffer_size):
    raw = simulate_ndarray((30, 40, 15), mask_rate = mask_rate)
    assert_identical_ndarrays(raw.var(), delayedarray.wrap(raw).var())
    assert_identical_ndarrays(raw.var(axis=0), delayedarray.wrap(raw).var(axis=0))

    y = delayedarray.wrap(raw) - 12
    ref = raw - 12
    assert numpy.isclose(ref.var(), y.var(buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=1), y.var(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=-1), y.var(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=(0, 2)), y.var(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_ndarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) + 29
    ref = raw + 29
    assert numpy.isclose(ref.var(), y.var(buffer_size=buffer_size))

    # Full masking is respected.
    y = delayedarray.wrap(numpy.ma.MaskedArray([1], mask=True)) + 20
    with pytest.warns(RuntimeWarning):
        assert y.var() is numpy.ma.masked

    # Zero-length array is respected.
    with pytest.warns(RuntimeWarning):
        y = delayedarray.wrap(numpy.ndarray((10, 0))) * 50
        assert numpy.isnan(y.var())


@pytest.mark.parametrize("mask_rate", [0, 0.5])
@pytest.mark.parametrize("buffer_size", [100, 500, 2000])
def test_SparseNdarray_var_sparse(mask_rate, buffer_size):
    raw = simulate_SparseNdarray((20, 30, 25), mask_rate = mask_rate)
    ref = raw * 19
    y = delayedarray.wrap(raw) * 19

    assert numpy.isclose(ref.var(), y.var(buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=1), y.var(axis=1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=-1), y.var(axis=-1, buffer_size=buffer_size))
    assert_close_ndarrays(ref.var(axis=(0, 2)), y.var(axis=(0, 2), buffer_size=buffer_size))

    # Trying with a single dimension.
    test_shape = (100,)
    raw = simulate_SparseNdarray((100,), mask_rate=mask_rate)
    y = delayedarray.wrap(raw) * 12
    ref = raw * 12
    assert numpy.isclose(ref.var(), y.var(buffer_size=buffer_size))

    # Full masking is respected.
    ref = delayedarray.SparseNdarray((1,), (numpy.zeros(1, dtype=numpy.int_), numpy.ma.MaskedArray([1], mask=True)))
    y = delayedarray.wrap(ref) / 5
    with pytest.warns(RuntimeWarning):
        assert y.var() is numpy.ma.masked

    # Zero-length array is respected.
    with pytest.warns(RuntimeWarning):
        y = delayedarray.wrap(delayedarray.SparseNdarray((0,), None, dtype=numpy.int32, index_dtype=numpy.int32)) * 50
        assert numpy.isnan(y.var())
