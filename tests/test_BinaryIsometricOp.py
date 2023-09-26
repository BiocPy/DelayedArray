import delayedarray
import numpy
import scipy.sparse


def test_BinaryIsometricOp_add():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert z.seed.left.shape == test_shape
    assert z.seed.right.shape == test_shape
    assert (numpy.array(z) == y + y2).all()


def test_BinaryIsometricOp_subtract():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - y2).all()


def test_BinaryIsometricOp_multiply():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - y2).all()


def test_BinaryIsometricOp_divide():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x / x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y / y2).all()


def test_BinaryIsometricOp_modulo():
    test_shape = (22, 44)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x % x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y % y2).all()


def test_BinaryIsometricOp_floordivide():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x // x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y // y2).all()


def test_BinaryIsometricOp_power():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x**x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y**y2).all()


def test_BinaryIsometricOp_equal():
    test_shape = (30, 55, 10)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x == x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y == y2)).all()


def test_BinaryIsometricOp_not_equal():
    test_shape = (12, 42)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x != x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y != y2)).all()


def test_BinaryIsometricOp_greater():
    test_shape = (42, 11)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x > x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y > y2)).all()


def test_BinaryIsometricOp_greater_equal():
    test_shape = (24, 13)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x >= x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y >= y2)).all()


def test_BinaryIsometricOp_less():
    test_shape = (24, 13)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x < x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y < y2)).all()


def test_BinaryIsometricOp_less_than():
    test_shape = (14, 33)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x <= x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y <= y2)).all()


def test_BinaryIsometricOp_logical_and():
    test_shape = (23, 33)
    y = numpy.random.rand(*test_shape) > 0.5
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_and(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_and(y, y2)).all()


def test_BinaryIsometricOp_logical_or():
    test_shape = (23, 55)
    y = numpy.random.rand(*test_shape) < 0.5
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_or(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_or(y, y2)).all()


def test_BinaryIsometricOp_logical_xor():
    test_shape = (44, 55)
    y = numpy.random.rand(*test_shape) < 0.5
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_xor(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_xor(y, y2)).all()


def test_BinaryIsometricOp_subset():
    test_shape = (44, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)

    z = x + x2
    ref = y + y2
    subset = (range(0, 44, 2), range(10, 50, 3))
    assert (delayedarray.extract_dense_array(z, subset) == ref[numpy.ix_(*subset)]).all()


def test_BinaryIsometricOp_sparse():
    y = scipy.sparse.random(100, 50, 0.1)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(100, 50)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_xor(x != 0, x2 != 0)
    assert not delayedarray.is_sparse(z)

    z = x + x2
    assert not delayedarray.is_sparse(z)

    y3 = scipy.sparse.random(100, 50, 0.1)
    x3 = delayedarray.DelayedArray(y3)
    z = x + x3
    assert delayedarray.is_sparse(z)


def test_BinaryIsometricOp_chunks():
    y = numpy.random.rand(20, 30)
    x = delayedarray.DelayedArray(y)
    z = x + x
    assert delayedarray.chunk_shape(z) == (1, 30)

    y2 = numpy.random.rand(30, 20).T
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2
    assert delayedarray.chunk_shape(z) == (20, 30)


def test_BinaryIsometricOp_dask():
    y = numpy.random.rand(20, 30)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(30, 20).T
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
