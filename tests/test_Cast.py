import numpy
import delayedarray


def test_Cast_simple():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    assert isinstance(z.seed, delayedarray.Cast)
    assert z.dtype == numpy.dtype("int32")
    assert z.shape == (30, 23)
    assert (numpy.array(z) == y.astype(numpy.int32)).all()
    assert delayedarray.chunk_shape(z) == (1, 23)


def test_Cast_dask():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)
    z = x.astype(numpy.int32)

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
