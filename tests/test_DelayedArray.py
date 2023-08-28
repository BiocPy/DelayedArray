import warnings

import delayedarray
import numpy


def test_DelayedArray_dense():
    raw = (numpy.random.rand(40, 30) * 5 - 10).astype(numpy.int32)
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'int32'") != -1

    dump = numpy.array(x)
    assert isinstance(dump, numpy.ndarray)
    assert (dump == raw).all()


def test_DelayedArray_isometric_add():
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


def test_DelayedArray_isometric_subtract():
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


def test_DelayedArray_isometric_multiply():
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


def test_DelayedArray_isometric_divide():
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


def test_DelayedArray_isometric_modulo():
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


def test_DelayedArray_isometric_floordivide():
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


def test_DelayedArray_isometric_power():
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


def test_DelayedArray_isometric_equal():
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


def test_DelayedArray_isometric_not_equal():
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


def test_DelayedArray_isometric_greater():
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


def test_DelayedArray_isometric_greater_equal():
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


def test_DelayedArray_isometric_less():
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


def test_DelayedArray_isometric_less_than():
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


def test_DelayedArray_isometric_logical_and():
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


def test_DelayedArray_isometric_logical_or():
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


def test_DelayedArray_isometric_logical_xor():
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


def test_DelayedArray_isometric_simple():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = -x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == -expanded).all()

    z = abs(x)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == abs(expanded)).all()

    for op in [
        "log",
        "log1p",
        "log2",
        "log10",
        "exp",
        "expm1",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "ceil",
        "floor",
        "trunc",
        "sign",
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ufunc = getattr(numpy, op)
            z = ufunc(x)
            obs = numpy.array(z)
            expected = ufunc(expanded)

        assert isinstance(z, delayedarray.DelayedArray)
        assert z.shape == x.shape
        assert z.seed.operation == op

        missing = numpy.isnan(obs)
        assert (missing == numpy.isnan(expected)).all()
        obs[missing] = 0
        expected[missing] = 0
        assert (obs == expected).all()


def test_DelayedArray_subset():
    test_shape = (30, 55, 20)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    sub = x[numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])]
    assert sub.shape == (9, 3, 4)
    assert isinstance(sub.seed.seed, numpy.ndarray)
    assert len(sub.seed.subset) == 3
    assert (
        numpy.array(sub) == y[numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])]
    ).all()

    # Works with slices for all (or all but one) dimensions.
    sub = x[0:15, 30:50, 0:20:2]
    assert sub.shape == (15, 20, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == y[0:15, 30:50, 0:20:2]).all()

    sub = x[:, :, range(0, 20, 2)]
    assert sub.shape == (30, 55, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == y[:, :, range(0, 20, 2)]).all()

    # Works with booleans.
    booled = [False] * test_shape[-1]
    booled[2] = True
    booled[3] = True
    booled[5] = True
    sub = x[:, :, booled]
    assert sub.shape == (30, 55, 3)
    assert (sub.seed.subset[-1] == numpy.array([2, 3, 5])).all()
    assert (numpy.array(sub) == y[:, :, booled]).all()

    # Works when fewer indices are supplied.
    sub = x[[1, 3, 5]]
    assert sub.shape == (3, 55, 20)
    assert (numpy.array(sub) == y[[1, 3, 5]]).all()

    sub = x[:, [1, 3, 5]]
    assert sub.shape == (30, 3, 20)
    assert (numpy.array(sub) == y[:, [1, 3, 5]]).all()

    # Falls back to a concrete numpy.ndarray
    stuff = x[:, :, 2]
    assert (stuff == y[:, :, 2]).all()


#    # Trying vectorized index.
#    stuff = x[[1,2,3],[4,5,6],[7,8,9]]
#    assert stuff.shape == (3,)


def test_DelayedArray_combine():
    y1 = delayedarray.DelayedArray(numpy.random.rand(30, 23))
    y2 = delayedarray.DelayedArray(numpy.random.rand(50, 23))
    x = numpy.concatenate((y1, y2))
    assert isinstance(x, delayedarray.DelayedArray)
    assert x.shape == (80, 23)
    assert x.dtype == numpy.float64
    assert x.seed.along == 0
    assert (numpy.array(x) == numpy.concatenate((y1.seed, y2.seed))).all()

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


def test_DelayedArray_transpose():
    y = numpy.random.rand(30, 23)
    x = delayedarray.DelayedArray(y)

    t = x.T
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert (numpy.array(t) == y.T).all()

    t = numpy.transpose(x)
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (23, 30)
    assert (numpy.array(t) == numpy.transpose(y)).all()

    # Adding more dimensions.
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


def test_DelayedArray_cast():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)

    z = x.astype(numpy.int32)
    assert isinstance(z.seed, delayedarray.Cast)
    assert z.dtype == numpy.int32
    assert z.shape == (30, 23)
    assert (numpy.array(z) == y.astype(numpy.int32)).all()


def test_DelayedArray_round():
    y = numpy.random.rand(30, 23) * 10
    x = delayedarray.DelayedArray(y)

    # Default.
    z = numpy.round(x)
    assert isinstance(z.seed, delayedarray.Round)
    assert z.dtype == numpy.float64
    assert z.shape == (30, 23)
    assert (numpy.array(z) == numpy.round(y)).all()

    # Number of places.
    z = numpy.round(x, decimals=1)
    assert (numpy.array(z) == numpy.round(y, decimals=1)).all()
