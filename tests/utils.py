from typing import List, Tuple
import numpy
import random
import delayedarray


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


def simulate_ndarray(shape: Tuple[int, ...], dtype: numpy.dtype = numpy.dtype("float64"), mask_rate: float = 0):
    y = numpy.random.rand(*shape)
    if isinstance(dtype, numpy.integer):
        y *= 10 # just to get some more interesting values
    y = y.astype(dtype, copy=False)
    if mask_rate:
        y = numpy.ma.MaskedArray(y, numpy.random.rand(*shape) < mask_rate)
    return y


def mock_SparseNdarray_contents(
    shape: Tuple[int, ...], 
    density1: float = 0.5, 
    density2: float = 0.5, 
    lower: float = -1, 
    upper: float = 1, 
    dtype: numpy.dtype = numpy.dtype("float64"), 
    mask_rate: float = 0
):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.uniform(lower, upper))

        new_indices = numpy.array(new_indices, dtype=numpy.int32)
        new_values = numpy.array(new_values, dtype=dtype)
        if mask_rate:
            new_mask = numpy.random.rand(len(new_values)) < mask_rate
            new_values = numpy.ma.MaskedArray(new_values, mask=new_mask)
        return new_indices, new_values

    # We use a survivor system to force at least one element of each dimension to 
    # proceed to the next recursion depth; this ensures that the type can be inferred.
    new_content = []
    survivor = random.randint(0, shape[-1] - 1)
    for i in range(shape[-1]):
        if i != survivor and random.uniform(0, 1) > density1:
            new_content.append(None)
        else:
            new_content.append(
                mock_SparseNdarray_contents(
                    shape[:-1],
                    density1=density1,
                    density2=density2,
                    lower=lower,
                    upper=upper,
                    dtype=dtype,
                    mask_rate=mask_rate,
                )
            )

    return new_content


def simulate_SparseNdarray(shape, **kwargs):
    contents = mock_SparseNdarray_contents(shape, **kwargs)
    return delayedarray.SparseNdarray(shape, contents)
