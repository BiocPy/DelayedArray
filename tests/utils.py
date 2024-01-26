from typing import List
import numpy


def sanitize_ndarray(x: numpy.ndarray):
    if numpy.ma.is_masked(x):
        if isinstance(x.mask, bool):
            if not x.mask:
                return x.data
        else:
            if not x.mask.any():
                return x.data
    return x


def assert_identical_ndarrays(x: numpy.ndarray, y: numpy.ndarray): 
    x = sanitize_ndarray(x)
    y = sanitize_ndarray(y)
    assert numpy.ma.is_masked(x) == numpy.ma.is_masked(y)
    if numpy.ma.is_masked(x):
        assert (x.mask == y.mask).all()
        comp = is_equal_with_nan(x.data, y.data)
        assert numpy.logical_or(x.mask, comp).all()
    else:
        assert is_equal_with_nan(x, y).all()


def is_equal_with_nan(left: numpy.ndarray, right: numpy.ndarray):
    if numpy.issubdtype(left.dtype, numpy.floating) or numpy.issubdtype(right.dtype, numpy.floating):
        return numpy.logical_or(numpy.isnan(left) == numpy.isnan(right), left == right)
    else:
        return left == right


def safe_concatenate(x: List[numpy.ndarray], axis: int = 0):
    if any(numpy.ma.is_masked(y) for y in x):
        return numpy.ma.concatenate(x, axis=axis)
    else:
        return numpy.concatenate(x, axis=axis)
