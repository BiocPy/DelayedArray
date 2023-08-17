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
    op = delayedarray.UnaryIsometricOpWithArgs(y, numpy.zeros(50), "+")
    opout = delayedarray.extract_dense_array(op, (slice(None), slice(None)))
    assert delayedarray.is_sparse(op)
    assert (opout == delayedarray.extract_dense_array(y, (slice(None), slice(None)))).all()

