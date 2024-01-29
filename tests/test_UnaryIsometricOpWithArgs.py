import delayedarray
import numpy
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, simulate_SparseNdarray


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_basics(mask_rate):
    test_shape = (55, 15)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x + 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert delayedarray.is_masked(z) == (mask_rate > 0)

    assert isinstance(z.seed.seed, numpy.ndarray)
    assert z.seed.right
    assert z.seed.operation == "add"
    assert z.seed.value == 2
    assert z.seed.along is None

    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + 2)
    assert delayedarray.chunk_shape(z) == (1, 15)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_add(left_mask_rate, right_mask_rate):
    test_shape = (55, 15)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x + 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert not delayedarray.is_sparse(z)
    assert delayedarray.is_masked(z) == (left_mask_rate > 0)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + 2)

    z = 5 + x
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + 5)

    v = simulate_ndarray((15,), mask_rate=right_mask_rate)
    z = x + v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert delayedarray.is_masked(z) == (left_mask_rate + right_mask_rate > 0)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + v)
    assert z.seed.along == 1

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v + x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v + y)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_subtract(left_mask_rate, right_mask_rate):
    test_shape = (55, 15)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x - 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y - 2)

    z = 5 - x
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5 - y)

    v = simulate_ndarray((15,), mask_rate=right_mask_rate)
    z = x - v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y - v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v - x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v - y)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_negate(mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    z = -x

    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert not delayedarray.is_sparse(z)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), -y)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_multiply(left_mask_rate, right_mask_rate):
    test_shape = (35, 25)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x * 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y * 2)

    z = 5 * x
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5 * y)

    v = simulate_ndarray((25,), mask_rate=right_mask_rate)
    z = x * v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y * v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v * x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v * y)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_divide(left_mask_rate, right_mask_rate):
    test_shape = (35, 25)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x / 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y / 2)

    z = 5 / (x + 1)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5 / (y + 1))

    v = simulate_ndarray((25,), mask_rate=right_mask_rate)
    z = x / v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y / v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v / (x + 1)
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v / (y + 1))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_modulo(left_mask_rate, right_mask_rate):
    test_shape = (22, 44)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x % 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y % 2)

    z = 5 % (x + 1)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5 % (y + 1))

    v = simulate_ndarray((44,), mask_rate=right_mask_rate)
    z = x % v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y % v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v % (x + 1)
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v % (y + 1))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_floordivide(left_mask_rate, right_mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x // 2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y // 2)

    z = 5 // (x + 1)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5 // (y + 1))

    v = simulate_ndarray((55,), mask_rate=right_mask_rate)
    z = x // v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y // v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v // (x + 1)
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v // (y + 1))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_power(left_mask_rate, right_mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x**2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert numpy.allclose(
        delayedarray.to_dense_array(z), y**2
    )  # guess if it's 2, it uses a special squaring, and the numeric precision changes.

    z = 5**x
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), 5**y)

    v = simulate_ndarray((55,), mask_rate=right_mask_rate)
    z = x**v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y**v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v**x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v**y)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_equal(left_mask_rate, right_mask_rate):
    test_shape = (30, 55, 10)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    val = x[0,0,0]
    z = x == val
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y == val))

    val = x[1,1,1]
    z = val == x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y == val))

    # Generating something that could actually compare equal for a proper test.
    v = y[2,3,:]
    if right_mask_rate:
        v = numpy.ma.MaskedArray(v, mask = numpy.random.rand(len(v)) < right_mask_rate)
    elif left_mask_rate:
        v = v.data

    z = x == v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (v == y))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v == x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v == y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_not_equal(left_mask_rate, right_mask_rate):
    test_shape = (12, 42)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    val = x[0,0]
    z = x != val
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y != val))

    val = x[1,1]
    z = val != x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y != val))

    # Generating something that could actually compare equal for a proper test.
    v = y[2,:]
    if right_mask_rate:
        v = numpy.ma.MaskedArray(v, mask = numpy.random.rand(len(v)) < right_mask_rate)
    elif left_mask_rate:
        v = v.data

    z = x != v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (v != y))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v != x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v != y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_greater(left_mask_rate, right_mask_rate):
    test_shape = (42, 11)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x > 0.5
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y > 0.5))

    z = 0.2 > x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y < 0.2))

    v = simulate_ndarray((11,), mask_rate=right_mask_rate)
    z = x > v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y > v))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v > x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v > y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_greater_equal(left_mask_rate, right_mask_rate):
    test_shape = (24, 13)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x >= 0.2
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y >= 0.2))

    z = 0.2 >= x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y <= 0.2))

    v = simulate_ndarray((13,), mask_rate=right_mask_rate)
    z = x >= v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y >= v))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v >= x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v >= y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_less(left_mask_rate, right_mask_rate):
    test_shape = (24, 13)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x < 0.8
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y < 0.8))

    z = 0.4 < x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y > 0.4))

    v = simulate_ndarray((13,), mask_rate=right_mask_rate)
    z = x < v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y < v))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v < x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v < y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_less_than(left_mask_rate, right_mask_rate):
    test_shape = (14, 33)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = x <= 0.7
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y <= 0.7))

    z = 0.2 <= x
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y >= 0.2))

    v = simulate_ndarray((33,), mask_rate=right_mask_rate)
    z = x <= v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), (y <= v))

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v <= x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), (v <= y))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_logical_and(left_mask_rate, right_mask_rate):
    test_shape = (23, 33)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_and(x, True)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_and(y, True))

    z = numpy.logical_and(False, x)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_and(y, False))

    v = simulate_ndarray((33,), dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    z = numpy.logical_and(v, x)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_and(v, y))

    z = numpy.logical_and(x, v)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_and(y, v))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_logical_or(left_mask_rate, right_mask_rate):
    test_shape = (23, 55)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_or(x, True)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_or(y, True))

    z = numpy.logical_or(False, x)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_or(y, False))

    v = simulate_ndarray((55,), dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    z = numpy.logical_or(v, x)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_or(v, y))

    z = numpy.logical_or(x, v)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_or(y, v))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_logical_xor(left_mask_rate, right_mask_rate):
    test_shape = (44, 55)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    z = numpy.logical_xor(x, True)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_xor(y, True))

    z = numpy.logical_xor(False, x)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_xor(y, False))

    v = simulate_ndarray((55,), dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    z = numpy.logical_xor(v, x)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_xor(v, y))

    z = numpy.logical_xor(x, v)
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.logical_xor(y, v))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_sparse(left_mask_rate, right_mask_rate):
    y = simulate_SparseNdarray((100, 50), density1=0.1, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)
    densed = delayedarray.to_dense_array(y)
    z = x + 1
    assert not delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), densed + 1)

    v = simulate_ndarray((50,), mask_rate=right_mask_rate)
    z = x / v
    assert delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), densed / v)

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v * x
        assert delayedarray.is_sparse(z)
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v * densed)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_with_array(left_mask_rate, right_mask_rate):
    y = simulate_ndarray((10, 20, 30), dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    v = simulate_ndarray((10, 1, 1), mask_rate=right_mask_rate)
    z = x + v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + v)
    assert z.seed.along == 0

    v = simulate_ndarray((1, 20, 1), mask_rate=right_mask_rate)
    z = x + v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + v)
    assert z.seed.along == 1

    v = simulate_ndarray((1, 1, 30), mask_rate=right_mask_rate)
    z = x + v
    assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.to_dense_array(z), y + v)
    assert z.seed.along == 2 

    if right_mask_rate == 0: # due to bug in MaskedArray ufunc dispatch.
        z = v * x
        assert isinstance(z.seed, delayedarray.UnaryIsometricOpWithArgs)
        assert z.shape == x.shape
        assert_identical_ndarrays(delayedarray.to_dense_array(z), v * y)
        assert z.seed.along == 2


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_subset(left_mask_rate, right_mask_rate):
    test_shape = (44, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)
    sub = (range(0, 40, 2), range(10, 50, 5))

    z = x + 1
    ref = y + 1
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub), ref[numpy.ix_(*sub)])

    v = simulate_ndarray((55,), mask_rate=right_mask_rate)
    z = x + v 
    ref = y + v
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub), ref[numpy.ix_(*sub)])

    v = simulate_ndarray((1, 55), mask_rate=right_mask_rate)
    z = x + v 
    ref = y + v
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub), ref[numpy.ix_(*sub)])

    v = simulate_ndarray((44, 1), mask_rate=right_mask_rate)
    z = x + v 
    ref = y + v
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, sub), ref[numpy.ix_(*sub)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_UnaryIsometricOpWithArgs_dask(mask_rate):
    y = simulate_ndarray((100, 50), mask_rate=mask_rate)
    x = delayedarray.DelayedArray(y)
    z = x + 1

    import dask.array
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.to_dense_array(z), da.compute())
