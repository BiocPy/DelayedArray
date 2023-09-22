import numpy
import delayedarray


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


def test_Combine_otherdim():
    y1 = delayedarray.DelayedArray(
        (numpy.random.rand(19, 43) * 100).astype(numpy.int32)
    )
    y2 = delayedarray.DelayedArray(
        (numpy.random.rand(19, 57) * 100).astype(numpy.int32)
    )

    x = numpy.concatenate((y1, y2), axis=1)
    assert isinstance(x, delayedarray.DelayedArray)
    assert x.shape == (19, 100)
    assert x.dtype == numpy.int32
    assert x.seed.along == 1
    assert (numpy.array(x) == numpy.concatenate((y1.seed, y2.seed), axis=1)).all()
    assert delayedarray.chunk_shape(x) == (1, 57)


def test_Combine_mixed():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(23, 50).astype(numpy.int32))
    x = numpy.concatenate((y1, y2.T))
    assert x.dtype == numpy.float64
    assert delayedarray.chunk_shape(x) == (50, 23)


def test_Combine_dask():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(50, 23))
    x = numpy.concatenate((y1, y2))

    import dask
    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(x) == da.compute()).all()


