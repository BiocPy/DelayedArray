import numpy
import delayedarray
import scipy


def test_Combine_simple():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(50, 23))
    x = numpy.concatenate((y1, y2))

    assert isinstance(x, delayedarray.DelayedArray)
    assert x.shape == (80, 23)
    assert x.dtype == numpy.float64
    assert x.seed.along == 0
    assert (numpy.array(x) == numpy.concatenate((y1.seed, y2.seed))).all()
    assert delayedarray.chunk_shape(x) == (1, 23)
    assert not delayedarray.is_sparse(x)


def test_Combine_otherdim():
    y1 = delayedarray.DelayedArray((numpy.random.rand(19, 43) * 100).astype(numpy.int32))
    y2 = delayedarray.DelayedArray((numpy.random.rand(19, 57) * 100).astype(numpy.int32))

    x = numpy.concatenate((y1, y2), axis=1)
    assert isinstance(x, delayedarray.DelayedArray)
    assert x.shape == (19, 100)
    assert x.dtype == numpy.int32
    assert x.seed.along == 1
    assert (numpy.array(x) == numpy.concatenate((y1.seed, y2.seed), axis=1)).all()
    assert delayedarray.chunk_shape(x) == (1, 57)


def test_Combine_subset():
    y1 = numpy.random.rand(30, 23)
    x1 = delayedarray.DelayedArray(y1)
    y2 = numpy.random.rand(50, 23)
    x2 = delayedarray.DelayedArray(y2)

    z = numpy.concatenate((x1, x2))
    ref = numpy.concatenate((y1, y2))
    subset = (range(5, 70, 2), range(3, 20))
    assert (delayedarray.extract_dense_array(z, subset) == ref[numpy.ix_(*subset)]).all()

    y2b = numpy.random.rand(30, 19)
    x2b = delayedarray.DelayedArray(y2b)
    z = numpy.concatenate((x1, x2b), axis=1)
    ref = numpy.concatenate((y1, y2b), axis=1)
    subset = (range(5, 28), range(10, 40))
    assert (delayedarray.extract_dense_array(z, subset) == ref[numpy.ix_(*subset)]).all()


def test_Combine_mixed_chunks():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(23, 50).astype(numpy.int32))
    x = numpy.concatenate((y1, y2.T))
    assert x.dtype == numpy.float64
    assert delayedarray.chunk_shape(x) == (50, 23)


def test_Combine_mixed_sparse():
    y1 = delayedarray.DelayedArray(scipy.sparse.rand(100, 10, 0.2))
    y2a = delayedarray.DelayedArray(scipy.sparse.rand(100, 20, 0.2))
    x = numpy.concatenate((y1, y2a), axis=1)
    assert delayedarray.is_sparse(x)

    y2b = numpy.random.rand(100, 20)
    x = numpy.concatenate((y1, y2b), axis=1)
    assert not delayedarray.is_sparse(x)


def test_Combine_dask():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(50, 23))
    x = numpy.concatenate((y1, y2))

    import dask
    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(x) == da.compute()).all()
