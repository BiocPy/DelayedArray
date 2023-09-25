import copy
import warnings

import delayedarray
import pytest
import numpy

#######################################################
#######################################################

import random
from delayedarray.SparseNdarray import _extract_dense_array_from_SparseNdarray, _extract_sparse_array_from_SparseNdarray

def mock_SparseNdarray_contents(
    shape, density1=0.5, density2=0.5, lower=-1, upper=1, dtype=numpy.float64
):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.uniform(lower, upper))
        if len(new_values):
            return numpy.array(new_indices), numpy.array(new_values, dtype=dtype)
        else:
            return None
    else:
        new_content = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density1:
                new_content.append(None)
            else:
                new_content.append(
                    mock_SparseNdarray_contents(
                        shape[1:],
                        density1=density1,
                        density2=density2,
                        lower=lower,
                        upper=upper,
                        dtype=dtype,
                    )
                )
        return new_content


def _recursive_compute_reference(contents, at, max_depth, triplets):
    if len(at) == max_depth - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                for j in range(len(idx)):
                    triplets.append(((*at, i, idx[j]), val[j]))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                _recursive_compute_reference(contents[i], (*at, i), max_depth, triplets)


def convert_SparseNdarray_to_numpy(x):
    contents = x._contents
    shape = x.shape
    triplets = []

    if len(shape) == 1:
        idx, val = contents
        for j in range(len(idx)):
            triplets.append(((idx[j],), val[j]))
    elif contents is not None:
        _recursive_compute_reference(contents, (), len(shape), triplets)

    output = numpy.zeros(shape)
    for pos, val in triplets:
        output[(..., *pos)] = val
    return output


def _compare_sparse_vectors(left, right):
    idx_l, val_l = left
    idx_r, val_r = right
    if len(idx_l) != len(idx_r):
        return False
    if not (idx_l == idx_r).all():
        return False
    if not (val_l == val_r).all():
        return False
    return True


def _recursive_compare_contents(left, right, at, max_depth):
    if len(left) != len(right):
        return False
    if len(at) == max_depth - 2:
        for i in range(len(left)):
            if left[i] is not None:
                if right[i] is None:
                    return False
                if not _compare_sparse_vectors(left[i], right[i]):
                    return False
    else:
        for i in range(len(left)):
            if left[i] is not None:
                if not _recursive_compare_contents(
                    left[i], right[i], (*at, i), max_depth
                ):
                    return False
    return True


def are_SparseNdarrays_equal(x, y):
    if x.shape != y.shape:
        return False
    maxdim = len(x.shape)
    contents1 = x._contents
    contents2 = y._contents

    if isinstance(contents1, list):
        if isinstance(contents2, list):
            return _recursive_compare_contents(contents1, contents2, (), maxdim)
        else:
            return False
    elif contents1 is None:
        if contents2 is None:
            return True
        else:
            return False
    else:
        return _compare_sparse_vectors(contents1, contents2)


def slices2ranges(slices, shape):
    output = []
    for i, s in enumerate(slices):
        output.append(range(*s.indices(shape[i])))
    return (*output,)

#######################################################
#######################################################


def test_SparseNdarray_check():
    test_shape = (10, 15, 20)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape
    assert y.dtype == numpy.float64

    with pytest.raises(ValueError, match="match the extent"):
        y = delayedarray.SparseNdarray((5, 15, 20), contents)

    with pytest.raises(ValueError, match="out of range"):
        y = delayedarray.SparseNdarray((10, 15, 1), contents)

    def scramble(con, depth):
        if depth == len(test_shape) - 2:
            for x in con:
                if x is not None:
                    i, v = x
                    random.shuffle(i)
        else:
            for x in con:
                if x is not None:
                    scramble(x, depth + 1)

    contents2 = copy.deepcopy(contents)
    scramble(contents2, 0)
    with pytest.raises(ValueError, match="should be sorted"):
        y = delayedarray.SparseNdarray(test_shape, contents2)

    def shorten(con, depth):
        if depth == len(test_shape) - 2:
            for i in range(len(con)):
                if con[i] is not None:
                    con[i] = (con[i][0][:-1], con[i][1])
        else:
            for x in con:
                if x is not None:
                    shorten(x, depth + 1)

    contents2 = copy.deepcopy(contents)
    shorten(contents2, 0)
    with pytest.raises(ValueError, match="should be the same"):
        y = delayedarray.SparseNdarray(test_shape, contents2)

    with pytest.raises(ValueError, match="Inconsistent data type"):
        y = delayedarray.SparseNdarray(test_shape, contents, dtype=numpy.int32)

    with pytest.raises(ValueError, match="cannot infer 'dtype'"):
        y = delayedarray.SparseNdarray(test_shape, None)

    empty = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.int32)
    assert empty.shape == test_shape
    assert empty.dtype == numpy.int32


#######################################################
#######################################################


def test_SparseNdarray_extract_dense_array_3d():
    test_shape = (16, 32, 8)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = numpy.array(y)
    assert (output == convert_SparseNdarray_to_numpy(y)).all()

    # Sliced extraction.
    slices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(10, 30), slice(None), slice(None))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()


def test_SparseNdarray_extract_dense_array_2d():
    test_shape = (50, 100)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    output = numpy.array(y)
    assert (output == convert_SparseNdarray_to_numpy(y)).all()

    # Sliced extraction.
    slices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(20, 30), slice(None))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(10, 80))
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()


def test_SparseNdarray_extract_dense_array_1d():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.dtype == numpy.float64

    # Full extraction.
    output = numpy.array(y)
    assert (output == convert_SparseNdarray_to_numpy(y)).all()

    # Sliced extraction.
    slices = (slice(5, 90, 7),)
    sliced = _extract_dense_array_from_SparseNdarray(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()


def test_SparseNdarray_extract_sparse_array_3d():
    test_shape = (20, 15, 10)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    full = [slice(None)] * len(test_shape)
    output = y[(*full,)]
    assert are_SparseNdarrays_equal(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(test_shape[0]), slice(0, 20, 2), slice(test_shape[2]))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(test_shape[0]), slice(test_shape[1]), slice(0, 8, 2))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(10, 30), slice(test_shape[1]), slice(test_shape[2]))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()


def test_SparseNdarray_extract_sparse_array_2d():
    test_shape = (99, 40)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    full = [slice(None)] * len(test_shape)
    output = y[(*full,)]
    assert are_SparseNdarrays_equal(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(20, 30), slice(None))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(None), slice(10, 80))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()


def test_SparseNdarray_extract_sparse_array_1d():
    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    # Full extraction.
    full = (slice(None),)
    output = y[(*full,)]
    assert are_SparseNdarrays_equal(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(5, 90, 7),)
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()


def test_SparseNdarray_int_type():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape
    assert y.dtype == numpy.int16

    full_indices = [range(d) for d in test_shape]
    dout = _extract_dense_array_from_SparseNdarray(y, full_indices)
    assert dout.dtype == numpy.int16
    ref = convert_SparseNdarray_to_numpy(y)
    assert (dout == ref).all()

    spout = _extract_sparse_array_from_SparseNdarray(y, full_indices)
    assert spout.dtype == numpy.int16
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()


def test_SparseNdarray_empty():
    test_shape = (20, 21, 22)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.uint32)
    assert y.shape == test_shape
    assert y.dtype == numpy.uint32

    full_indices = [range(d) for d in test_shape]
    dout = _extract_dense_array_from_SparseNdarray(y, full_indices)
    assert (dout == numpy.zeros(test_shape)).all()
    dout = _extract_dense_array_from_SparseNdarray(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert (dout == numpy.zeros((3, 4, 5))).all()

    spout = _extract_sparse_array_from_SparseNdarray(y, full_indices)
    assert spout._contents is None
    assert spout.shape == test_shape
    assert spout.dtype == numpy.uint32
    spout = _extract_sparse_array_from_SparseNdarray(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert spout.shape == (3, 4, 5)


def test_SparseNdarray_0d():
    y = delayedarray.SparseNdarray((), None, dtype=numpy.uint32)
    assert y.shape == ()
    assert y.dtype == numpy.uint32

    y = delayedarray.SparseNdarray((), 5, dtype=numpy.uint32)
    assert y.shape == ()
    assert y.dtype == numpy.uint32

    with pytest.raises(ValueError, match="0-dimensional"):
        y = delayedarray.SparseNdarray((), {}, dtype=numpy.uint32)


#######################################################
#######################################################


def test_SparseNdarray_abs():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    out = abs(y)
    assert (numpy.array(out) == abs(numpy.array(y))).all()

    # Checking that the transformer does something sensible here.
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.float64)
    out = abs(y)
    assert (numpy.array(out) == numpy.zeros(test_shape)).all()

    test_shape = (99,)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    out = abs(y)
    assert (numpy.array(out) == abs(numpy.array(y))).all()


def test_SparseNdarray_neg():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = -y
    assert (numpy.array(out) == -ref).all()


def test_SparseNdarray_ufunc_simple():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = numpy.log1p(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert out.dtype == numpy.float32
    assert (numpy.array(out) == numpy.log1p(ref)).all()

    out = numpy.exp(y)
    assert isinstance(out, numpy.ndarray)
    assert out.dtype == numpy.float32
    assert (out == numpy.exp(ref)).all()


#######################################################
#######################################################


def test_SparseNdarray_add():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1 + y
    assert isinstance(out, numpy.ndarray)
    assert (out == 1 + ref).all()
    out = y + 2
    assert isinstance(out, numpy.ndarray)
    assert (out == ref + 2).all()

    other = numpy.random.rand(40)
    out = other + y
    assert isinstance(out, numpy.ndarray)
    assert (out == other + ref).all()
    out = y + other
    assert isinstance(out, numpy.ndarray)
    assert (out == ref + other).all()

    other = numpy.random.rand(30, 1)
    out = other + y
    assert isinstance(out, numpy.ndarray)
    assert (out == other + ref).all()
    out = y + other 
    assert isinstance(out, numpy.ndarray)
    assert (out == ref + other).all()


def test_SparseNdarray_sub():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 - y
    assert isinstance(out, numpy.ndarray)
    assert (out == 1.5 - ref).all()
    out = y - 2.5
    assert isinstance(out, numpy.ndarray)
    assert (out == ref - 2.5).all()

    other = numpy.random.rand(40)
    out = other - y
    assert isinstance(out, numpy.ndarray)
    assert (out == other - ref).all()
    out = y - other
    assert isinstance(out, numpy.ndarray)
    assert (out == ref - other).all()

    other = numpy.random.rand(30, 1)
    out = other - y
    assert isinstance(out, numpy.ndarray)
    assert (out == other - ref).all()
    out = y - other 
    assert isinstance(out, numpy.ndarray)
    assert (out == ref - other).all()


def test_SparseNdarray_multiply():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == 1.5 * ref).all()
    out = y * 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref * 2).all()

    other = numpy.random.rand(40)
    out = other * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == other * ref).all()
    out = y * other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref * other).all()

    other = numpy.random.rand(30, 1)
    out = other * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == other * ref).all()
    out = y * other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref * other).all()


def test_SparseNdarray_divide():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 / y
        assert isinstance(out, numpy.ndarray)
        assert (out == 1.5 / ref).all()
    out = y / 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref / 2).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other / y
        assert isinstance(out, numpy.ndarray)
        assert (out == other / ref).all()
    out = y / other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref / other).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other / y
        assert isinstance(out, numpy.ndarray)
        assert (out == other / ref).all()
    out = y / other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref / other).all()


def test_SparseNdarray_floor_divide():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 // y
        assert isinstance(out, numpy.ndarray)
        assert (out == 1.5 // ref).all()
    out = y // 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref // 2).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other // y
        assert isinstance(out, numpy.ndarray)
        assert (out == other // ref).all()
    out = y // other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref // other).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other // y
        assert isinstance(out, numpy.ndarray)
        assert (out == other // ref).all()
    out = y // other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref // other).all()


def test_SparseNdarray_modulo():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    def equal_with_nan(left, right):
        missing = numpy.isnan(left)
        assert (missing == numpy.isnan(right)).all()
        left[missing] = 0
        right[missing] = 0
        assert (left == right).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 % y
        assert isinstance(out, numpy.ndarray)
        equal_with_nan(out, 1.5 % ref)
    out = y % 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % 2).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        equal_with_nan(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % other).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        equal_with_nan(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % other).all()


def test_SparseNdarray_power():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 ** y
    assert isinstance(out, numpy.ndarray)
    assert (out == 1.5 ** ref).all()
    out = y ** 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref ** 2).all()

    other = numpy.random.rand(40)
    out = other ** y
    assert isinstance(out, numpy.ndarray)
    assert (out == other ** ref).all()
    out = y ** other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref ** other).all()

    other = numpy.random.rand(30, 1)
    out = other ** y
    assert isinstance(out, numpy.ndarray)
    assert (out == other ** ref).all()
    out = y ** other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref ** other).all()


def test_SparseNdarray_equal():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (1.5 == ref)).all()
    out = y == 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref == 2)).all()

    other = numpy.random.rand(40)
    out = other == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other == ref)).all()
    out = y == other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref == other)).all()

    other = numpy.random.rand(30, 1)
    out = other == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other == ref)).all()
    out = y == other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref == other)).all()


def test_SparseNdarray_not_equal():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 != y
    assert isinstance(out, numpy.ndarray)
    assert (out == (1.5 != ref)).all()
    out = y != 2
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref != 2)).all()

    other = numpy.random.rand(40)
    out = other != y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other != ref)).all()
    out = y != other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref != other)).all()

    other = numpy.random.rand(30, 1)
    out = other != y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other != ref)).all()
    out = y != other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref != other)).all()


def test_SparseNdarray_greater_than_or_equal():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 >= y
    assert isinstance(out, numpy.ndarray)
    assert (out == (1.5 >= ref)).all()
    out = y >= 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref >= 2)).all()

    other = numpy.random.rand(40)
    out = other >= y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other >= ref)).all()
    out = y >= other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref >= other)).all()

    other = numpy.random.rand(30, 1)
    out = other >= y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other >= ref)).all()
    out = y >= other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref >= other)).all()


def test_SparseNdarray_greater():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 > y
    assert isinstance(out, numpy.ndarray)
    assert (out == (1.5 > ref)).all()
    out = y > 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref > 2)).all()

    other = numpy.random.rand(40)
    out = other > y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other > ref)).all()
    out = y > other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref > other)).all()

    other = numpy.random.rand(30, 1)
    out = other > y
    assert isinstance(out, numpy.ndarray)
    assert (out == (other > ref)).all()
    out = y > other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref > other)).all()


def test_SparseNdarray_less_than_or_equal():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (1.5 <= ref)).all()
    out = y <= 2
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref <= 2)).all()

    other = numpy.random.rand(40)
    out = other <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other <= ref)).all()
    out = y <= other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref <= other)).all()

    other = numpy.random.rand(30, 1)
    out = other <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other <= ref)).all()
    out = y <= other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref <= other)).all()


def test_SparseNdarray_less_than_or_equal():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=1, upper=10, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = 1.5 < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (1.5 < ref)).all()
    out = y < 2
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref < 2)).all()

    other = numpy.random.rand(40)
    out = other < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other < ref)).all()
    out = y < other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref < other)).all()

    other = numpy.random.rand(30, 1)
    out = other < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (other < ref)).all()
    out = y < other
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref < other)).all()


#######################################################
#######################################################


def test_SparseNdarray_astype():
    test_shape = (50, 30, 20)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)

    z = y.astype(numpy.float64)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert (numpy.array(z) == numpy.array(y)).all()


def test_SparseNdarray_round():
    test_shape = (50, 30, 20)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.float64)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    z = numpy.round(y)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert (numpy.array(z) == numpy.round(ref)).all()

    z = numpy.round(y, decimals=1)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert (numpy.array(z) == numpy.round(ref, decimals=1)).all()
