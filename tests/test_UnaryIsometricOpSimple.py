import warnings

import delayedarray
import numpy
import scipy


def test_UnaryIsometricOpSimple_basic():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    expanded = numpy.array(x)

    import dask
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
            da = delayedarray.create_dask_array(z).compute()
            expected = ufunc(expanded)

        assert isinstance(z, delayedarray.DelayedArray)
        assert z.shape == x.shape
        assert z.seed.operation == op
        assert delayedarray.chunk_shape(z) == (1, 55)
        assert not delayedarray.is_sparse(z)

        missing = numpy.isnan(obs)
        assert (missing == numpy.isnan(expected)).all()
        assert (missing == numpy.isnan(da)).all()
        obs[missing] = 0
        expected[missing] = 0
        da[missing] = 0
        assert (obs == expected).all()
        assert (obs == da).all()


def test_UnaryIsometricOpSimple_negate():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    z = -x

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape

    expanded = numpy.array(x)
    assert (numpy.array(z) == -expanded).all()


def test_UnaryIsometricOpSimple_logical_not():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape) > 0.5
    x = delayedarray.DelayedArray(y)
    z = numpy.logical_not(x)

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape

    expanded = numpy.array(x)
    assert (numpy.array(z) == numpy.logical_not(expanded)).all()


def test_UnaryIsometricOpSimple_abs():
    test_shape = (30, 55)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)

    # Absolute values have their own dunder method, so we check it explicitly.
    z = abs(x)

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape

    expanded = numpy.array(x)
    assert (numpy.array(z) == abs(expanded)).all()


def test_UnaryIsometricOpSimple_subset():
    test_shape = (40, 65)
    y = numpy.random.rand(*test_shape)
    x = delayedarray.DelayedArray(y)
    z = abs(x)

    ref = abs(y)
    sub = (range(0, 40, 2), range(0, 60, 3))
    assert (delayedarray.extract_dense_array(z, sub) == ref[numpy.ix_(*sub)]).all()


def test_UnaryIsometricOpSimple_sparse():
    y = scipy.sparse.rand(20, 50, 0.5)
    x = delayedarray.DelayedArray(y)

    z = numpy.exp(x)
    assert not delayedarray.is_sparse(z)
    assert (numpy.exp(y.toarray()) == numpy.array(z)).all()

    z = numpy.log1p(x)
    assert delayedarray.is_sparse(z)
    assert (numpy.log1p(y.toarray()) == numpy.array(z)).all()
