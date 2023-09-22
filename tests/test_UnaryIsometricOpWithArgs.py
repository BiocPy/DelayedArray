import delayedarray
import numpy
import dask.array


def test_UnaryIsometricOpWithArgs_isometric_add():
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

    z = 5 + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + 5).all()

    v = numpy.random.rand(15)
    z = v + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v + y).all()
    assert delayedarray.chunk_shape(z) == (1, 15)

    v = numpy.random.rand(15)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 1

    v = numpy.random.rand(55, 1)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y + v).all()
    assert z.seed.along == 0

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert z.seed.left.shape == test_shape
    assert z.seed.right.shape == test_shape
    assert (numpy.array(z) == y + y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_subtract():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_multiply():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y - y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_divide():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x / x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y / y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_modulo():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x % x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y % y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_floordivide():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x // x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y // y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_power():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x**x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == y**y2).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_equal():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x == x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y == y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_not_equal():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x != x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y != y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_greater():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x > x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y > y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_greater_equal():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x >= x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y >= y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_less():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x < x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y < y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_less_than():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = x <= x2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == (y <= y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_logical_and():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_and(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_and(y, y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_logical_or():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_or(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_or(y, y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()


def test_UnaryIsometricOpWithArgs_isometric_logical_xor():
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

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_xor(x, x2)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == numpy.logical_xor(y, y2)).all()

    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert (numpy.array(z) == da.compute()).all()
