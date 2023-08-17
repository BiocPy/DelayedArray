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

#
#def test_UnaryIsometricOpWithArgs_addition():
#    test_shape = (10, 15, 20)
#    contents = mock_SparseNdarray_contents(test_shape)
#    y = delayedarray.SparseNdarray(test_shape, contents)
#
#
#
#
#    assert y.shape == test_shape
#
#
