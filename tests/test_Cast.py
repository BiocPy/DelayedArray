import numpy
import delayedarray
import scipy.sparse


def test_Cast_simple():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    assert isinstance(z.seed, delayedarray.Cast)
    assert z.dtype == numpy.dtype("int32")
    assert z.shape == (30, 23)
    assert (numpy.array(z) == y.astype(numpy.int32)).all()
    assert delayedarray.chunk_shape(z) == (1, 23)
    assert not delayedarray.is_sparse(z)


def test_Cast_subset():
    test_shape = (30, 20)
    y = numpy.random.rand(*test_shape) * 10
    x = delayedarray.DelayedArray(y)

    z = x.astype(numpy.int32)
    ref = y.astype(numpy.int32)
    subset = (range(10, 20), range(5, 15))
    assert (delayedarray.extract_dense_array(z, subset) == ref[numpy.ix_(*subset)]).all()


def test_Cast_sparse():
    y = scipy.sparse.random(10, 20, 0.05)
    x = delayedarray.DelayedArray(y)

    z = x.astype(numpy.int32)
    assert delayedarray.is_sparse(z)


def test_Cast_dask():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
