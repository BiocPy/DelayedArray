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

    t = x.T
    assert isinstance(t.seed, delayedarray.Transpose)
    assert t.shape == (30, 40)
    assert (numpy.array(t) == dump.T).all()


def test_DelayedArray_isometric_add():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x + 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert isinstance(z.seed.seed, numpy.ndarray)
    assert z.seed.right
    assert z.seed.operation == "+"
    assert z.seed.value == 2
    assert z.seed.along is None
    assert (numpy.array(z) == expanded + 2).all()

    z = 5 + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + 5).all()

    v = numpy.random.rand(15)
    z = v + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v + expanded).all()

    v = numpy.random.rand(15)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + v).all()
    assert z.seed.along == 1

    v = numpy.random.rand(55, 1)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + v).all()
    assert z.seed.along == 0


def test_DelayedArray_isometric_subtract():
    test_shape = (55, 15)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x - 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded - 2).all()

    z = 5 - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 - expanded).all()

    v = numpy.random.rand(15)
    z = v - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v - expanded).all()

    z = x - v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded - v).all()


def test_DelayedArray_isometric_multiply():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x * 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded * 2).all()

    z = 5 * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 * expanded).all()

    v = numpy.random.rand(25)
    z = v * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v * expanded).all()

    z = x * v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded * v).all()


def test_DelayedArray_isometric_divide():
    test_shape = (35, 25)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x / 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded / 2).all()

    z = 5 / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 / (expanded + 1)).all()

    v = numpy.random.rand(25)
    z = v / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v / (expanded + 1)).all()

    z = x / v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded / v).all()


def test_DelayedArray_isometric_modulo():
    test_shape = (22, 44)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x % 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded % 2).all()

    z = 5 % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 % (expanded + 1)).all()

    v = numpy.random.rand(44)
    z = v % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v % (expanded + 1)).all()

    z = x % v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded % v).all()


def test_DelayedArray_isometric_floordivide():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x // 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded // 2).all()

    z = 5 // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 // (expanded + 1)).all()

    v = numpy.random.rand(55)
    z = v // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v // (expanded + 1)).all()

    z = x // v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded // v).all()


def test_DelayedArray_isometric_power():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x**2
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded**2).all()

    z = 5**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5**expanded).all()

    v = numpy.random.rand(55)
    z = v**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == v**expanded).all()

    z = x**v
    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded**v).all()


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

    sub = x[slice(1, 10), [20, 30, 40], [10, 11, 12, 13]]
    assert sub.shape == (9, 3, 4)
    assert isinstance(sub.seed.seed, numpy.ndarray)
    assert len(sub.seed.subset) == 3
    assert (
        numpy.array(sub)
        == numpy.array(x)[numpy.ix_(range(1, 10), [20, 30, 40], [10, 11, 12, 13])]
    ).all()

    sub = x[:, :, range(0, 20, 2)]
    assert sub.shape == (30, 55, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == numpy.array(x)[:, :, range(0, 20, 2)]).all()


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
