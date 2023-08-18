import random
import delayedarray
import pytest
import copy
import numpy
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

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(15), "*", along = 1) 
    assert delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    assert op.shape == test_shape
    assert not delayedarray.is_sparse(op)

    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(20), "+", along = 2) 
    assert not delayedarray.is_sparse(op)

    with pytest.raises(ValueError, match="should be non-negative"):
        op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(20), "+", along = -1) 

    with pytest.raises(ValueError, match="length of array 'value'"):
        op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.random.rand(20), "+", along = 0) 


def test_UnaryIsometricOpWithArgs_scalar_addition():
    test_shape = (50, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    
    # Full extraction.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    assert not delayedarray.is_sparse(op)
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert (opout == delayedarray.extract_dense_array(y, (slice(None), slice(None))) + 5).all()

    # Partial extraction
    op = delayedarray.UnaryIsometricOpWithArgs(y, 5, "+")
    opout = delayedarray.extract_dense_array(op, (slice(2, 50), slice(0, 30)))
    assert (opout == delayedarray.extract_dense_array(y, (slice(2, 50), slice(0, 30))) + 5).all()

    # Adding zero.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 0, "+")
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert delayedarray.is_sparse(op)
    assert (opout == delayedarray.extract_dense_array(y, (slice(None), slice(None)))).all()


def test_UnaryIsometricOpWithArgs_vector_addition():
    test_shape = (50, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    
    # Full extraction.
    v = numpy.random.rand(40)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+", along=1)
    assert not delayedarray.is_sparse(op)
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert (opout == delayedarray.extract_dense_array(y, (slice(None), slice(None))) + v).all()

    # Partial extraction
    v = numpy.random.rand(50)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+")
    opout = delayedarray.extract_dense_array(op, (slice(10, 40), slice(0, 30)))
    ref = delayedarray.extract_dense_array(y, (slice(10, 40), slice(0, 30)))
    assert (opout == (ref.T + v[10:40]).T).all()

    v = numpy.random.rand(40)
    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "+", along=1)
    opout = delayedarray.extract_dense_array(op, (slice(10, 40), slice(5, 30)))
    assert (opout == delayedarray.extract_dense_array(y, (slice(10, 40), slice(5, 30))) + v[5:30]).all()

    # Adding zero.
    noop = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(50), "+")
    assert delayedarray.is_sparse(noop)
    opout = delayedarray.extract_dense_array(noop, (slice(None), slice(None)))
    assert (opout == delayedarray.extract_dense_array(y, (slice(None), slice(None)))).all()


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
    assert (convert_SparseNdarray_to_numpy(spout._contents, spout.shape) == dout).all()

    # Partial extraction
    op = delayedarray.UnaryIsometricOpWithArgs(y, -2, "*")
    indices = (slice(2, 10), slice(0, 14, 2), slice(None))

    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == delayedarray.extract_dense_array(y, indices) * -2).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert isinstance(spout, delayedarray.SparseNdarray)
    assert (convert_SparseNdarray_to_numpy(spout._contents, spout.shape) == dout).all()

    # Multiplying by one.
    op = delayedarray.UnaryIsometricOpWithArgs(y, 1, "*")

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices)).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert are_SparseNdarray_contents_equal(spout._contents, y._contents, len(test_shape))

    # Multiplying by some non-finite value.
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.NaN, "*")
    opout = delayedarray.extract_dense_array(op, full_indices)
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
    assert (convert_SparseNdarray_to_numpy(spout._contents, spout.shape) == dout).all()

    # Partial extraction
    v = numpy.random.rand(15)
    indices = (slice(2, 18), slice(0, 15, 2), slice(None))

    ref = delayedarray.extract_dense_array(y, indices)
    my_indices = range(*indices[1].indices(15))
    for i in range(len(my_indices)):
        ref[:,i,:] *= v[my_indices[i]]

    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*", along=1)
    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert (convert_SparseNdarray_to_numpy(spout._contents, spout.shape) == ref).all()

    # Another partial extraction
    v = numpy.random.rand(20)
    indices = (slice(10, 20), slice(None), slice(0, 10, 2))

    ref = delayedarray.extract_dense_array(y, indices)
    my_indices = range(*indices[0].indices(20))
    for i in range(len(my_indices)):
        ref[i,:,:] *= v[my_indices[i]]

    op = delayedarray.UnaryIsometricOpWithArgs(y, v, "*", along=0)
    dout = delayedarray.extract_dense_array(op, indices)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(op, indices)
    assert (convert_SparseNdarray_to_numpy(spout._contents, spout.shape) == ref).all()

    # Multiplying by one.
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.ones(10), "*", along=2)

    dout = delayedarray.extract_dense_array(op, full_indices)
    assert (dout == delayedarray.extract_dense_array(y, full_indices)).all()

    spout = delayedarray.extract_sparse_array(op, full_indices)
    assert are_SparseNdarray_contents_equal(spout._contents, y._contents, len(y.shape))

    # Multiplying by a bad number.
    bad = numpy.zeros(10)
    bad[5] = numpy.NaN
    op = delayedarray.UnaryIsometricOpWithArgs(y, bad, "*", along=2)
    assert not delayedarray.is_sparse(op)
