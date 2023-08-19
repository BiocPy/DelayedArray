import warnings

import delayedarray
import numpy
import pytest
from utils import *


def test_UnaryIsometricOpWithArgs_check():
    test_shape = (10, 15, 20)
    y = numpy.random.rand(*test_shape)
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    assert op.shape == test_shape
    assert not delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(10), "+")
    assert not delayedarray.is_sparse(op)

    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "*")
    assert op.shape == test_shape
    assert delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(15), "*", along=1)
    assert delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    assert op.shape == test_shape
    assert not delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(20), "+", along=2)
    assert not delayedarray.is_sparse(op)

    with pytest.raises(ValueError, match="should be non-negative"):
        op = delayedarray.UnaryIsometricOpWithArgs(
            y, numpy.random.rand(20), "+", along=-1
        )

    with pytest.raises(ValueError, match="length of array 'value'"):
        op = delayedarray.UnaryIsometricOpWithArgs(
            y, numpy.random.rand(20), "+", along=0
        )


###############################################################
## We'll use addition as an exemplar of a dense-only operation.


def test_UnaryIsometricOpWithArgs_scalar_addition():
    test_shape = (50, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    assert not delayedarray.is_sparse(op)
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert (
        opout == delayedarray.extract_dense_array(y, (slice(None), slice(None))) + 5
    ).all()

    # Partial extraction
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    opout = delayedarray.extract_dense_array(op, (slice(2, 50), slice(0, 30)))
    assert (
        opout == delayedarray.extract_dense_array(y, (slice(2, 50), slice(0, 30))) + 5
    ).all()

    # Adding zero.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "+")
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert delayedarray.is_sparse(op)
    assert (
        opout == delayedarray.extract_dense_array(y, (slice(None), slice(None)))
    ).all()


def test_UnaryIsometricOpWithArgs_vector_addition():
    test_shape = (50, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    v = numpy.random.rand(40)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+", along=1)
    assert not delayedarray.is_sparse(op)
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert (
        opout == delayedarray.extract_dense_array(y, (slice(None), slice(None))) + v
    ).all()

    # Partial extraction
    v = numpy.random.rand(50)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+")
    opout = delayedarray.extract_dense_array(op, (slice(10, 40), slice(0, 30)))
    ref = delayedarray.extract_dense_array(y, (slice(10, 40), slice(0, 30)))
    assert (opout == (ref.T + v[10:40]).T).all()

    v = numpy.random.rand(40)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+", along=1)
    opout = delayedarray.extract_dense_array(op, (slice(10, 40), slice(5, 30)))
    assert (
        opout
        == delayedarray.extract_dense_array(y, (slice(10, 40), slice(5, 30))) + v[5:30]
    ).all()

    # Adding zero.
    noop = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(50), "+")
    assert delayedarray.is_sparse(noop)
    opout = delayedarray.extract_dense_array(noop, (slice(None), slice(None)))
    assert (
        opout == delayedarray.extract_dense_array(y, (slice(None), slice(None)))
    ).all()


def test_UnaryIsometricOpWithArgs_1d_addition():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape, density1=0)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 9, "+")
    opout = delayedarray.extract_dense_array(op, (slice(None),))
    assert (opout == delayedarray.extract_dense_array(y, (slice(None),)) + 9).all()

    # Partial extraction.
    v = numpy.random.rand(test_shape[0])
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+")
    opout = delayedarray.extract_dense_array(op, (slice(10, 40),))
    ref = delayedarray.extract_dense_array(y, (slice(10, 40),))
    assert (opout == ref + v[10:40]).all()


###############################################################################
# Simlarly, we'll use multiplication as an exemplar of a sparse-only operation.


def test_UnaryIsometricOpWithArgs_scalar_multiplication():
    test_shape = (20, 15, 10)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "*")
    assert delayedarray.is_sparse(op)

    full_indices = (slice(None), slice(None), slice(None))
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices) * 5).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert isinstance(spout, delayedarray.SparseNdarray)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Partial extraction
    op = delayedarray.UnaryIsometricOpWithArgs(y, -2, "*")
    indices = (slice(2, 10), slice(0, 14, 2), slice(None))

    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == delayedarray.extract_dense_array(y, indices) * -2).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert isinstance(spout, delayedarray.SparseNdarray)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Multiplying by one.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 1, "*")

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices)).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert are_SparseNdarrays_equal(spout, y)

    # Multiplying by some non-finite value.
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.NaN, "*")
    delayedarray.extract_dense_array(op, full_indices)
    assert not delayedarray.is_sparse(op)


def test_UnaryIsometricOpWithArgs_vector_multiplication():
    test_shape = (20, 15, 10)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    v = numpy.random.rand(10)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*", along=2)
    assert delayedarray.is_sparse(op)

    full_indices = (slice(None), slice(None), slice(None))
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices) * v).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert isinstance(spout, delayedarray.SparseNdarray)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Partial extraction
    v = numpy.random.rand(15)
    indices = (slice(2, 18), slice(0, 15, 2), slice(None))

    ref = delayedarray.extract_dense_array(y, indices)
    my_indices = range(*indices[1].indices(15))
    for i in range(len(my_indices)):
        ref[:, i, :] *= v[my_indices[i]]

    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*", along=1)
    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()

    # Another partial extraction
    v = numpy.random.rand(20)
    indices = (slice(10, 20), slice(None), slice(0, 10, 2))

    ref = delayedarray.extract_dense_array(y, indices)
    my_indices = range(*indices[0].indices(20))
    for i in range(len(my_indices)):
        ref[i, :, :] *= v[my_indices[i]]

    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*", along=0)
    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()

    # Multiplying by one.
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.ones(10), "*", along=2)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices)).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert are_SparseNdarrays_equal(spout, y)

    # Multiplying by a bad number.
    bad = numpy.zeros(10)
    bad[5] = numpy.NaN
    op = delayedarray.UnaryIsometricOpWithArgs(y, bad, "*", along=2)
    assert not delayedarray.is_sparse(op)


def test_UnaryIsometricOpWithArgs_1d_multiplication():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape, density1=0)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 9, "*")
    dout = delayedarray.extract_dense_array(op, (slice(None),))
    assert (dout == delayedarray.extract_dense_array(y, (slice(None),)) * 9).all()

    spout = delayedarray.extract_sparse_array(op, (slice(None),))
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Partial extraction.
    v = numpy.random.rand(test_shape[0])
    ref = delayedarray.extract_dense_array(y, (slice(10, 40),)) * v[10:40]

    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*")
    dout = delayedarray.extract_dense_array(op, (slice(10, 40),))
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, (slice(10, 40),))
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()


##############################################################
# For the remaining operations, we just do some cursory tests.


def test_UnaryIsometricOpWithArgs_subtraction():
    test_shape = (10, 11, 12)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    full_indices = (slice(None), slice(None), slice(None))
    ref = delayedarray.extract_dense_array(y, full_indices)

    # Scalar
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "-")
    assert not delayedarray.is_sparse(op)
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref - 5).all()

    # Vector
    v = numpy.random.rand(12)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "-", right=False, along=2)
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == v - ref).all()

    # All-zeros.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "-", right=True, along=2)
    assert delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(10), "-", right=False)
    assert delayedarray.is_sparse(op)
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == 0 - ref).all()


def test_UnaryIsometricOpWithArgs_division():
    test_shape = (20, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    full_indices = (slice(None), slice(None))
    ref = delayedarray.extract_dense_array(y, full_indices)

    # Scalar; skipping warnings due to division by zero.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "/", right=False)
        assert not delayedarray.is_sparse(op)

        dout = delayedarray.extract_dense_array(op, full_indices)
        assert (dout == 5 / ref).all()

    # Vector
    v = numpy.random.rand(20)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "/")
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == (ref.T / v).T).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector, dividing on the left.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, v, "/", right=False)
        assert not delayedarray.is_sparse(op)

        dout = delayedarray.extract_dense_array(op, full_indices)
        assert (dout == (v / ref.T).T).all()

    # Any zeros.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "/")
        assert not delayedarray.is_sparse(op)

        op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(20), "/")
        assert not delayedarray.is_sparse(op)

    # All ones.
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.ones(30), "/", along=1)
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()


def test_UnaryIsometricOpWithArgs_modulo():
    test_shape = (22, 17, 15)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    full_indices = (slice(None), slice(None), slice(None))
    ref = delayedarray.extract_dense_array(y, full_indices)

    # Scalar.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0.2, "%")
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref % 0.2).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector
    v = numpy.random.rand(15)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "%", along=2)
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref % v).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector, dividing on the left.
    v = numpy.random.rand(22)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, v, "%", right=False)
        assert not delayedarray.is_sparse(op)

        dout = delayedarray.extract_dense_array(op, full_indices)
        computed = (v % ref.T).T

        # Dealing with the incomparability of IEEE NaN's by replacing them with
        # a placeholder for comparison purposes.
        isnan = numpy.isnan(computed)
        assert (isnan == numpy.isnan(dout)).all()
        dout[isnan] = -1234
        computed[isnan] = -1234
        assert (dout == computed).all()

    # Any zeros.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "%")
        assert not delayedarray.is_sparse(op)

        op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(22), "%")
        assert not delayedarray.is_sparse(op)


def test_UnaryIsometricOpWithArgs_floordivision():
    test_shape = (22, 57)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    full_indices = (slice(None), slice(None))
    ref = delayedarray.extract_dense_array(y, full_indices)

    # Scalar.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0.1, "//")
    assert delayedarray.is_sparse(op)
    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref // 0.1).all()

    # Vector
    v = numpy.random.rand(22)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "//")
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == (ref.T // v).T).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector, dividing on the left.
    v = numpy.random.rand(57)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, v, "//", right=False, along=1)
        assert not delayedarray.is_sparse(op)

        dout = delayedarray.extract_dense_array(op, full_indices)
        assert (dout == v // ref).all()

    # Any zeros.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "//")
        assert not delayedarray.is_sparse(op)

        op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(22), "//")
        assert not delayedarray.is_sparse(op)


def test_UnaryIsometricOpWithArgs_power():
    test_shape = (42, 37)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    full_indices = (slice(None), slice(None))
    ref = delayedarray.extract_dense_array(y, full_indices)

    # Scalar.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 2, "**")
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == ref**2).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector. This needs rounding to avoid NaNs when taking fractional powers of negatives.
    # We also add 1 to avoid zeros that break sparsity from 0 ** 0.
    v = numpy.round(numpy.random.rand(42) * 3) + 1
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "**")
    assert delayedarray.is_sparse(op)

    dout = delayedarray.extract_dense_array(op, full_indices)
    obs = (ref.T**v).T
    assert numpy.allclose(dout, obs) # some kind of numeric difference between ** and **=, perhaps. Who knows, but exact comparison sometimes fails.

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert (convert_SparseNdarray_to_numpy(spout) == dout).all()

    # Vector, on the left.
    v = numpy.round(numpy.random.rand(37) * 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        op = delayedarray.UnaryIsometricOpWithArgs(y, v, "**", right=False, along=1)
        assert not delayedarray.is_sparse(op)

        dout = delayedarray.extract_dense_array(op, full_indices)
        assert (dout == v**ref).all()

    # Any zeros.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "**")
    assert not delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(42), "**", right=False)
    assert not delayedarray.is_sparse(op)
