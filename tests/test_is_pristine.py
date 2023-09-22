import delayedarray
import numpy


def test_is_pristine():
    y = numpy.random.rand(100, 10)
    assert delayedarray.is_pristine(y)

    x = delayedarray.DelayedArray(y)
    assert delayedarray.is_pristine(x)

    z = x + 1
    assert not delayedarray.is_pristine(z)

    z = x[10:90:5,:]
    assert not delayedarray.is_pristine(z)
