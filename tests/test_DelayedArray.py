import warnings

import delayedarray
from utils import *
import numpy


def test_DelayedArray_dense():
    raw = (numpy.random.rand(40, 30) * 5 - 10).astype(numpy.int32)
    x = delayedarray.DelayedArray(raw)
    assert x.shape == raw.shape
    assert x.dtype == raw.dtype
    assert not delayedarray.is_sparse(x)

    out = str(x)
    assert out.find("<40 x 30> DelayedArray object of type 'int32'") != -1

    dump = numpy.array(x)
    assert isinstance(dump, numpy.ndarray)
    assert (dump == raw).all()


def test_DelayedArray_sparse():
    test_shape = (10, 20, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    assert x.shape == test_shape
    assert x.dtype == numpy.float64
    assert delayedarray.is_sparse(x)

    out = str(x)
    assert out.find("<10 x 20 x 30> sparse DelayedArray object of type 'float64'") != -1

    dump = numpy.array(x)
    assert isinstance(dump, numpy.ndarray)
    ref = convert_SparseNdarray_to_numpy(y)
    assert (dump == ref).all()

    out = delayedarray.extract_sparse_array(x, (slice(None), slice(None), slice(None)))
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (convert_SparseNdarray_to_numpy(out) == ref).all()


def test_DelayedArray_isometric_add():
    test_shape = (55, 15)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x + 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + 2).all()

    z = 5 + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + 5).all()

    v = numpy.random.rand(15)
    z = v + x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v + expanded).all()

    v = numpy.random.rand(15)
    z = x + v
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded + v).all()


def test_DelayedArray_isometric_subtract():
    test_shape = (55, 15)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x - 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded - 2).all()

    z = 5 - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 - expanded).all()

    v = numpy.random.rand(15)
    z = v - x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v - expanded).all()

    z = x - v
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded - v).all()


def test_DelayedArray_isometric_multiply():
    test_shape = (35, 25)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x * 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded * 2).all()

    z = 5 * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 * expanded).all()

    v = numpy.random.rand(25)
    z = v * x
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v * expanded).all()

    z = x * v
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded * v).all()


def test_DelayedArray_isometric_divide():
    test_shape = (35, 25)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x / 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded / 2).all()

    z = 5 / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 / (expanded + 1)).all()

    v = numpy.random.rand(25)
    z = v / (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v / (expanded + 1)).all()

    z = x / v
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded / v).all()


def test_DelayedArray_isometric_modulo():
    test_shape = (22, 44)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x % 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded % 2).all()

    z = 5 % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 % (expanded + 1)).all()

    v = numpy.random.rand(44)
    z = v % (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v % (expanded + 1)).all()

    z = x % v
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded % v).all()


def test_DelayedArray_isometric_floordivide():
    test_shape = (30, 55)
    contents = mock_SparseNdarray_contents(test_shape, lower=10, upper=20)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x // 2
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded // 2).all()

    z = 5 // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5 // (expanded + 1)).all()

    v = numpy.random.rand(55)
    z = v // (x + 1)
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v // (expanded + 1)).all()

    z = x // v
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded // v).all()


def test_DelayedArray_isometric_floordivide():
    test_shape = (30, 55)
    contents = mock_SparseNdarray_contents(test_shape, lower=0.1, upper=2)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = x**2
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded**2).all()

    z = 5**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == 5**expanded).all()

    v = numpy.random.rand(55)
    z = v**x
    assert isinstance(z, delayedarray.DelayedArray)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == v**expanded).all()

    z = x**v
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == expanded**v).all()


def test_DelayedArray_isometric_simple():
    test_shape = (30, 55)
    contents = mock_SparseNdarray_contents(test_shape, lower=-5, upper=5)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    z = -x
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert (numpy.array(z) == -expanded).all()

    z = abs(x)
    assert isinstance(z, delayedarray.DelayedArray)
    assert delayedarray.is_sparse(z)
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
            preserves_sparse = ufunc(0) != 0

        assert isinstance(z, delayedarray.DelayedArray)
        assert z.shape == x.shape
        assert delayedarray.is_sparse(z) or preserves_sparse

        missing = numpy.isnan(obs)
        assert (missing == numpy.isnan(expected)).all()
        obs[missing] = 0
        expected[missing] = 0
        assert (obs == expected).all()


def test_DelayedArray_subset():
    test_shape = (30, 55, 20)
    contents = mock_SparseNdarray_contents(test_shape, lower=-5, upper=5)
    y = delayedarray.SparseNdarray(test_shape, contents)
    x = delayedarray.DelayedArray(y)

    sub = x[2, [20, 30, 40], [10, 11, 12, 13]]
    assert sub.shape == (3, 4)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (
        numpy.array(sub)
        == numpy.array(x)[2, :, :][numpy.ix_([20, 30, 40], [10, 11, 12, 13])]
    ).all()

    sub = x[:, :, range(0, 20, 2)]
    assert sub.shape == (30, 55, 10)
    assert isinstance(sub._seed, delayedarray.Subset)
    assert (numpy.array(sub) == numpy.array(x)[:, :, range(0, 20, 2)]).all()
