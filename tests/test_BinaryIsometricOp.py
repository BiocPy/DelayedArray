import delayedarray
import numpy
import scipy.sparse
import pytest

from utils import simulate_ndarray, assert_identical_ndarrays, inject_mask_for_sparse_matrix


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_add(left_mask_rate, right_mask_rate):
    test_shape = (55, 15)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert z.seed.left.shape == test_shape
    assert z.seed.right.shape == test_shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y + y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_subtract(left_mask_rate, right_mask_rate):
    test_shape = (55, 15)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y - y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_multiply(left_mask_rate, right_mask_rate):
    test_shape = (35, 25)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x - x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y - y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_divide(left_mask_rate, right_mask_rate):
    test_shape = (35, 25)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x / x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y / y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_modulo(left_mask_rate, right_mask_rate):
    test_shape = (22, 44)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x % x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y % y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_floordivide(left_mask_rate, right_mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x // x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y // y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_power(left_mask_rate, right_mask_rate):
    test_shape = (30, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x**x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y**y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_equal(left_mask_rate, right_mask_rate):
    test_shape = (30, 55, 10)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x == x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y == y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_not_equal(left_mask_rate, right_mask_rate):
    test_shape = (12, 42)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x != x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y != y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_greater(left_mask_rate, right_mask_rate):
    test_shape = (42, 11)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x > x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y > y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_greater_equal(left_mask_rate, right_mask_rate):
    test_shape = (24, 13)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x >= x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y >= y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_less(left_mask_rate, right_mask_rate):
    test_shape = (24, 13)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x < x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y < y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_less_than(left_mask_rate, right_mask_rate):
    test_shape = (14, 33)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x <= x2

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y <= y2)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_logical_and(left_mask_rate, right_mask_rate):
    test_shape = (23, 33)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_and(x, x2)

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), numpy.logical_and(y, y2))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_logical_or(left_mask_rate, right_mask_rate):
    test_shape = (23, 55)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_or(x, x2)

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), numpy.logical_or(y, y2))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_logical_xor(left_mask_rate, right_mask_rate):
    test_shape = (44, 55)
    y = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, dtype=numpy.dtype("bool"), mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_xor(x, x2)

    assert isinstance(z, delayedarray.DelayedArray)
    assert z.shape == x.shape
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), numpy.logical_xor(y, y2))


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_subset(left_mask_rate, right_mask_rate):
    test_shape = (44, 55)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = numpy.random.rand(*test_shape)
    x2 = delayedarray.DelayedArray(y2)

    z = x + x2
    ref = y + y2
    subset = (range(0, 44, 2), range(10, 50, 3))
    assert_identical_ndarrays(delayedarray.extract_dense_array(z, subset), ref[numpy.ix_(*subset)])


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_sparse(left_mask_rate, right_mask_rate):
    y = scipy.sparse.random(100, 50, 0.1)
    inject_mask_for_sparse_matrix(y, left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(y.shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = numpy.logical_xor(x != 0, x2 != 0)
    assert not delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), numpy.logical_xor(y.toarray() != 0, y2 != 0))

    z = x + x2
    assert not delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y.toarray() + y2)

    y3 = scipy.sparse.random(100, 50, 0.1)
    x3 = delayedarray.DelayedArray(y3)
    z = x + x3
    assert delayedarray.is_sparse(z)
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), y.toarray() + y3)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_chunks(left_mask_rate, right_mask_rate):
    y = simulate_ndarray((20, 30), mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)
    z = x + x
    assert delayedarray.chunk_shape(z) == (1, 30)

    y2 = simulate_ndarray((30, 20), mask_rate=right_mask_rate).T # i.e., Fortran order.
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2
    assert delayedarray.chunk_shape(z) == (20, 30)


@pytest.mark.parametrize("left_mask_rate", [0, 0.2])
@pytest.mark.parametrize("right_mask_rate", [0, 0.2])
def test_BinaryIsometricOp_dask(left_mask_rate, right_mask_rate):
    test_shape = (20, 30)
    y = simulate_ndarray(test_shape, mask_rate=left_mask_rate)
    x = delayedarray.DelayedArray(y)

    y2 = simulate_ndarray(test_shape, mask_rate=right_mask_rate)
    x2 = delayedarray.DelayedArray(y2)
    z = x + x2

    import dask
    da = delayedarray.create_dask_array(z)
    assert isinstance(da, dask.array.core.Array)
    assert_identical_ndarrays(delayedarray.extract_dense_array(z), da.compute())
