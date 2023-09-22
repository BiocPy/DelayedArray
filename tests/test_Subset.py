import delayedarray
import numpy


def test_Subset_ix():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    sub = x[numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])]
    assert sub.shape == (9, 3, 4)
    assert isinstance(sub.seed.seed, numpy.ndarray)
    assert len(sub.seed.subset) == 3
    assert (
        numpy.array(sub) == y[numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])]
    ).all()
    assert delayedarray.chunk_shape(sub) == (1, 1, 4)


def test_Subset_slice():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    # Works with slices for all dimensions.
    sub = x[0:15, 30:50, 0:20:2]
    assert sub.shape == (15, 20, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == y[0:15, 30:50, 0:20:2]).all()

    # All but one dimension.
    sub = x[:, :, range(0, 20, 2)]
    assert sub.shape == (30, 55, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == y[:, :, range(0, 20, 2)]).all()


def test_Subset_booleans():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    booled = [False] * test_shape[-1]
    booled[2] = True
    booled[3] = True
    booled[5] = True
    sub = x[:, :, booled]
    assert sub.shape == (30, 55, 3)
    assert (sub.seed.subset[-1] == numpy.array([2, 3, 5])).all()
    assert (numpy.array(sub) == y[:, :, booled]).all()


def test_Subset_fewer_indices():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    # Works when fewer indices are supplied.
    sub = x[[1, 3, 5]]
    assert sub.shape == (3, 55, 20)
    assert (numpy.array(sub) == y[[1, 3, 5]]).all()

    sub = x[:, [1, 3, 5]]
    assert sub.shape == (30, 3, 20)
    assert (numpy.array(sub) == y[:, [1, 3, 5]]).all()


def test_Subset_unsorted_duplicates():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    sub = x[:, :, [1, 1, 2, 3]]
    assert (numpy.array(sub) == y[:, :, [1, 1, 2, 3]]).all()

    sub = x[:, [5, 4, 3, 2, 1, 0], :]
    assert (numpy.array(sub) == y[:, [5, 4, 3, 2, 1, 0], :]).all()


def test_Subset_collapse():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    stuff = x[:, :, 2]
    assert (stuff == y[:, :, 2]).all()
    stuff = x[0, :, 2]
    assert (stuff == y[0, :, 2]).all()

#    # Trying vectorized index.
#    stuff = x[[1,2,3],[4,5,6],[7,8,9]]
#    assert stuff.shape == (3,)


def test_Subset_dask():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    import dask
    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(x) == da.compute()).all()
