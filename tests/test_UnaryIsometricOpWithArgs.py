import delayedarray
import numpy
import scipy.sparse


def test_UnaryIsometricOpWithArgs_basics():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x + 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape

    assert isinstance(z.seed.seed, numpy.ndarray)
    assert z.seed.right
    assert z.seed.operation == "add"
    assert z.seed.value == 2
    assert z.seed.along is None

    assert (numpy.array(z) == y + 2).all()
    assert delayedarray.chunk_shape(z) == (1, 15)


def test_UnaryIsometricOpWithArgs_add():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x + 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert (numpy.array(z) == y + 2).all()

    z = 5 + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + 5).all()

    v = numpy.random.rand(15)
    z = v + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v + y).all()

    v = numpy.random.rand(15)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 1


def test_UnaryIsometricOpWithArgs_subtract():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x - 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - 2).all()

    z = 5 - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 - y).all()

    v = numpy.random.rand(15)
    z = v - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v - y).all()

    z = x - v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - v).all()


def test_UnaryIsometricOpWithArgs_multiply():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x * 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y * 2).all()

    z = 5 * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 * y).all()

    v = numpy.random.rand(25)
    z = v * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v * y).all()

    z = x * v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y * v).all()


def test_UnaryIsometricOpWithArgs_divide():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x / 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y / 2).all()

    z = 5 / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 / (y + 1)).all()

    v = numpy.random.rand(25)
    z = v / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v / (y + 1)).all()

    z = x / v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y / v).all()


def test_UnaryIsometricOpWithArgs_modulo():
    test_shape = (22, 44)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x % 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y % 2).all()

    z = 5 % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 % (y + 1)).all()

    v = numpy.random.rand(44)
    z = v % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v % (y + 1)).all()

    z = x % v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y % v).all()


def test_UnaryIsometricOpWithArgs_floordivide():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x // 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y // 2).all()

    z = 5 // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 // (y + 1)).all()

    v = numpy.random.rand(55)
    z = v // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v // (y + 1)).all()

    z = x // v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y // v).all()


def test_UnaryIsometricOpWithArgs_power():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x**2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert numpy.allclose(
        numpy.array(z), y**2
    )  # guess if it's 2, it uses a special squaring, and the numeric precision changes.

    z = 5**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5**y).all()

    v = numpy.random.rand(55)
    z = v**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v**y).all()

    z = x**v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y**v).all()


def test_UnaryIsometricOpWithArgs_equal():
    test_shape = (30, 55, 10)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x == 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y == 2)).all()

    z = 2 == x
    assert (numpy.array(z) == (y == 2)).all()

    v = numpy.random.rand(10)
    z = v == x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v == y)).all()


def test_UnaryIsometricOpWithArgs_not_equal():
    test_shape = (12, 42)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x != 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y != 2)).all()

    z = 2 != x
    assert (numpy.array(z) == (y != 2)).all()

    v = numpy.random.rand(42)
    z = v != x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v != y)).all()


def test_UnaryIsometricOpWithArgs_greater():
    test_shape = (42, 11)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x > 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y > 2)).all()

    z = 2 > x
    assert (numpy.array(z) == (y < 2)).all()

    v = numpy.random.rand(11)
    z = v > x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v > y)).all()


def test_UnaryIsometricOpWithArgs_greater_equal():
    test_shape = (24, 13)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x >= 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y >= 2)).all()

    z = 2 >= x
    assert (numpy.array(z) == (y <= 2)).all()

    v = numpy.random.rand(13)
    z = v >= x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v >= y)).all()


def test_UnaryIsometricOpWithArgs_less():
    test_shape = (24, 13)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x < 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y < 2)).all()

    z = 2 < x
    assert (numpy.array(z) == (y > 2)).all()

    v = numpy.random.rand(13)
    z = v < x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v < y)).all()


def test_UnaryIsometricOpWithArgs_less_than():
    test_shape = (14, 33)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    z = x <= 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y <= 2)).all()

    z = 2 <= x
    assert (numpy.array(z) == (y >= 2)).all()

    v = numpy.random.rand(33)
    z = v <= x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (v <= y)).all()


def test_UnaryIsometricOpWithArgs_logical_and():
    test_shape = (23, 33)
    y = numpy.random.rand(*test_shape) > 0.5
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_and(x, True)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_and(y, True)).all()

    z = numpy.logical_and(False, x)
    assert (numpy.array(z) == numpy.logical_and(y, False)).all()

    v = numpy.random.rand(33) > 0.5
    z = numpy.logical_and(v, x)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_and(v, y)).all()


def test_UnaryIsometricOpWithArgs_logical_or():
    test_shape = (23, 55)
    y = numpy.random.rand(*test_shape) < 0.5
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_or(x, True)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_or(y, True)).all()

    z = numpy.logical_or(False, x)
    assert (numpy.array(z) == numpy.logical_or(y, False)).all()

    v = numpy.random.rand(55) > 0.5
    z = numpy.logical_or(v, x)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_or(v, y)).all()


def test_UnaryIsometricOpWithArgs_logical_xor():
    test_shape = (44, 55)
    y = numpy.random.rand(*test_shape) < 0.5
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_xor(x, True)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_xor(y, True)).all()

    z = numpy.logical_xor(False, x)
    assert (numpy.array(z) == numpy.logical_xor(y, False)).all()

    v = numpy.random.rand(55) > 0.5
    z = numpy.logical_xor(v, x)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_xor(v, y)).all()


def test_UnaryIsometricOpWithArgs_sparse():
    y = scipy.sparse.random(100, 50, 0.1)
    x = delayedarray.DelayedArray(y)
    z = x + 1
    assert not delayedarray.is_sparse(z)
    assert (numpy.array(z) == y.toarray() + 1).all()

    v = numpy.random.rand(50)
    z = v * x
    assert delayedarray.is_sparse(z)
    assert (numpy.array(z) == v * y.toarray()).all()

    v = numpy.random.rand(50)
    z = x / v
    assert delayedarray.is_sparse(z)
    assert (numpy.array(z) == y.toarray() / v).all()


def test_UnaryIsometricOpWithArgs_with_array():
    y = numpy.random.rand(10, 20, 30) < 0.5
    x = delayedarray.DelayedArray(y)

    v = numpy.random.rand(10, 1, 1)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 0

    v = numpy.random.rand(1, 20, 1)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 1

    v = numpy.random.rand(1, 1, 30)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 2 


def test_UnaryIsometricOpWithArgs_subset():
    test_shape = (44, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    sub = (range(0, 40, 2), range(10, 50, 5))

    z = x + 1
    ref = y + 1
    assert (delayedarray.extract_dense_array(z, sub) == ref[numpy.ix_(*sub)]).all()

    v = numpy.random.rand(55)
    z = x + v 
    ref = y + v
    assert (delayedarray.extract_dense_array(z, sub) == ref[numpy.ix_(*sub)]).all()

    v = numpy.random.rand(1, 55)
    z = x + v 
    ref = y + v
    assert (delayedarray.extract_dense_array(z, sub) == ref[numpy.ix_(*sub)]).all()

    v = numpy.random.rand(44, 1)
    z = x + v 
    ref = y + v
    assert (delayedarray.extract_dense_array(z, sub) == ref[numpy.ix_(*sub)]).all()


def test_UnaryIsometricOpWithArgs_dask():
    y = numpy.random.rand(100, 50)
    x = delayedarray.DelayedArray(y)
    z = x + 1

    import dask.array
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
