import numpy
import dask.array
import delayedarray


def test_ndarray_default():
    raw = (numpy.random.rand(40, 30) * 5 - 10).astype(numpy.int32)
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'int32'") != -1

    dump = numpy.array(x)
    assert isinstance(dump, numpy.ndarray)
    assert (dump == raw).all()

    da = delayedarray.create_dask_array(x)
    assert isinstance(da, dask.array.core.Array)
    assert (dump == da.compute()).all()

    assert delayedarray.chunk_shape(x) == (1, 30)


def test_ndarray_colmajor():
    raw = numpy.random.rand(30, 40).T
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype
    assert delayedarray.chunk_shape(x) == (40, 1)

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'float64'") != -1
