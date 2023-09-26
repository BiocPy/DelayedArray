import numpy
import delayedarray
import scipy.sparse


def test_Round_default():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    assert isinstance(z.seed, delayedarray.Round)
    assert z.dtype == numpy.float64
    assert z.shape == (30, 23)
    assert (numpy.array(z) == numpy.round(y)).all()
    assert delayedarray.chunk_shape(z) == (1, 23)
    assert not delayedarray.is_sparse(z)


def test_Round_subset():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    ref = numpy.round(y)
    subset = (range(5, 20), range(3, 19, 2))
    assert (delayedarray.extract_dense_array(ref, subset) == ref[numpy.ix_(*subset)]).all()


def test_Round_sparse():
    y = scipy.sparse.random(30, 10, 0.05)
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)
    assert delayedarray.is_sparse(x)


def test_Round_decimals():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x, decimals=1)
    assert (numpy.array(z) == numpy.round(y, decimals=1)).all()


def test_Round_dask():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
