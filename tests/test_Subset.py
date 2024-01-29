import delayedarray
import numpy
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_ix(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    subix = numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])
    sub = x[subix]
    assert sub.shape == (9, 3, 4)
    assert isinstance(sub.seed.seed, numpy.ndarray)
    assert len(sub.seed.subset) == 3
    assert delayedarray.chunk_shape(sub) == (1, 1, 4)
    assert not delayedarray.is_sparse(sub)
    assert delayedarray.is_masked(sub) == (mask_rate > 0)

    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[subix])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_slice(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    # Works with slices for all dimensions.
    sub = x[0:15, 30:50, 0:20:2]
    assert sub.shape == (15, 20, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[0:15, 30:50, 0:20:2])

    # All but one dimension.
    sub = x[:, :, range(0, 20, 2)]
    assert sub.shape == (30, 55, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[:, :, range(0, 20, 2)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_booleans(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    booled = [False] * test_shape[-1]
    booled[2] = True
    booled[3] = True
    booled[5] = True
    sub = x[:, :, booled]

    assert sub.shape == (30, 55, 3)
    assert (sub.seed.subset[-1] == numpy.array([2, 3, 5])).all()
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[:, :, booled])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_fewer_indices(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    # Works when fewer indices are supplied.
    sub = x[[1, 3, 5]]
    assert sub.shape == (3, 55, 20)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[[1, 3, 5]])

    sub = x[:, [1, 3, 5]]
    assert sub.shape == (30, 3, 20)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[:, [1, 3, 5]])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_unsorted_duplicates(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    sub = x[:, :, [1, 1, 2, 3]]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[:, :, [1, 1, 2, 3]])

    sub = x[:, [5, 4, 3, 2, 1, 0], :]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), y[:, [5, 4, 3, 2, 1, 0], :])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_subset(mask_rate):
    y = simulate_ndarray((99, 63), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    sub1 = (slice(5, 70, 2), slice(3, 20))
    z = x[sub1]
    ref = y[sub1]

    sub2 = (range(2, 20), range(2, 18, 2))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub2), ref[numpy.ix_(*sub2)])
    sub2 = (range(ref.shape[0]), range(2, 18, 2))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub2), ref[numpy.ix_(*sub2)])
    sub2 = (range(2, 20), range(ref.shape[1]))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub2), ref[numpy.ix_(*sub2)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_collapse(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    stuff = x[:, :, 2]
    assert_identical_ndarrays(stuff, y[:, :, 2])
    stuff = x[0, :, 2]
    assert_identical_ndarrays(stuff, y[0, :, 2])

#    # Trying vectorized index.
#    stuff = x[[1,2,3],[4,5,6],[7,8,9]]
#    assert stuff.shape == (3,)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_sparse(mask_rate):
    y = simulate_SparseNdarray((50, 20), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)

    sub = x[5:45:5, 0:20:2]
    assert delayedarray.is_sparse(sub)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), densed[5:45:5, 0:20:2])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_Subset_dask(mask_rate):
    test_shape = (30, 55, 20)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    sub = x[0:10:2,5:50:5,2:5]

    import dask
    da = delayedarray.create_dask_array(sub)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), da.compute())
