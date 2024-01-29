import warnings

import delayedarray
import numpy
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpSimple_basic(mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

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
            obs = delayedarray.to_dense_array(z)
            da = delayedarray.create_dask_array(z).compute()
            expected = ufunc(y)

        assert isinstance(z.seed, delayedarray.UnaryIsometricOpSimple)
        assert z.shape == x.shape
        assert z.seed.operation == op
        assert delayedarray.chunk_shape(z) == (1, 55)
        assert not delayedarray.is_sparse(z)
        assert delayedarray.is_masked(z) == (mask_rate > 0)

        assert_identical_ndarrays(obs, expected)
        assert_identical_ndarrays(obs, da)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpSimple_logical_not(mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    z = numpy.logical_not(x)

    assert isinstance(z.seed, delayedarray.UnaryIsometricOpSimple)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_not(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpSimple_abs(mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    # Absolute values have their own dunder method, so we check it explicitly.
    z = abs(x)

    assert isinstance(z.seed, delayedarray.UnaryIsometricOpSimple)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), abs(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpSimple_subset(mask_rate):
    test_shape = (40, 65)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    z = abs(x)

    ref = abs(y)
    sub = (range(0, 40, 2), range(0, 60, 3))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub), ref[numpy.ix_(*sub)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpSimple_sparse(mask_rate):
    y = simulate_SparseNdarray((20, 50), density1=0.15, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)

    z = numpy.exp(x)
    assert not delayedarray.is_sparse(z)
    assert_identical_ndarrays(numpy.exp(densed), delayedarray.to_dense_array(z))

    z = numpy.log1p(x)
    assert delayedarray.is_sparse(z)
    assert_identical_ndarrays(numpy.log1p(densed), delayedarray.to_dense_array(z))
