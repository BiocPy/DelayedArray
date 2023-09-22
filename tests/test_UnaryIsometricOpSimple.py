import warnings

import delayedarray
import numpy


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
