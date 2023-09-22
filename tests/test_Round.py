import numpy
import delayedarray


def test_Round_default():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = numpy.round(x)

    assert isinstance(z.seed, delayedarray.Round)
    assert z.dtype == numpy.float64
    assert z.shape == (30, 23)
    assert (numpy.array(z) == numpy.round(y)).all()
    assert delayedarray.chunk_shape(z) == (1, 23)


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
