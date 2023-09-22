import delayedarray
import numpy


def test_Transpose_simple():
    y = numpy.random.rand(30, 23)
    x = delayedarray.DelayedArray(y)

    t = x.T
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert (numpy.array(t) == y.T).all()
    assert delayedarray.chunk_shape(t) == (23, 1)

    t = numpy.transpose(x)
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert (numpy.array(t) == numpy.transpose(y)).all()


def test_Transpose_more_dimensions():
    y = numpy.random.rand(30, 23, 10)
    x = delayedarray.DelayedArray(y)

    t = numpy.transpose(x, axes=(1, 2, 0))
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 10, 30)
    assert (numpy.array(t) == numpy.transpose(y, axes=(1, 2, 0))).all()

    t = numpy.transpose(x)
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (10, 23, 30)
    assert (numpy.array(t) == numpy.transpose(y)).all()


def test_Transpose_dask():
    y = numpy.random.rand(30, 23, 10)
    x = delayedarray.DelayedArray(y)
    t = numpy.transpose(x)

    import dask
    da = delayedarray.create_dask_array(t)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(t) == da.compute()).all()
