import copy
import warnings

import delayedarray
import pytest
import numpy

#######################################################
#######################################################

import random

def mock_SparseNdarray_contents(shape, density1=0.5, density2=0.5, lower=-1, upper=1, dtype=numpy.float64):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.uniform(lower, upper))
        return numpy.array(new_indices, dtype=numpy.int32), numpy.array(new_values, dtype=dtype)

    # We use a survivor system to force at least one element of each dimension to 
    # proceed to the next recursion depth; this ensures that the type can be inferred.
    new_content = []
    survivor = random.randint(0, shape[-1])
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
                )
            )

    return new_content


def _recursive_compute_reference(contents, ndim, triplets, at = []):
    if len(at) == ndim - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                for j, ix in enumerate(idx):
                    triplets.append(((ix, i, *reversed(at)), val[j]))
    else:
        at.append(0)
        for i, con in enumerate(contents):
            if con is not None:
                at[-1] = i
                _recursive_compute_reference(con, ndim, triplets, at=at)
        at.pop()


def convert_SparseNdarray_to_numpy(x):
    contents = x._contents
    shape = x.shape
    triplets = []

    ndim = len(shape)
    if ndim == 1:
        idx, val = contents
        for j in range(len(idx)):
            triplets.append(((idx[j],), val[j]))
    elif contents is not None:
        _recursive_compute_reference(contents, ndim, triplets)

    output = numpy.zeros(shape)
    for pos, val in triplets:
        output[pos] = val
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


def _recursive_compare_contents(left, right, dim):
    if len(left) != len(right):
        return False
    if dim == 1:
        for i, lcon in enumerate(left):
            if lcon is not None:
                if right[i] is None:
                    return False
                if not _compare_sparse_vectors(lcon, right[i]):
                    return False
    else:
        for i, lcon in enumerate(left):
            if lcon is not None:
                if not _recursive_compare_contents(lcon, right[i], dim - 1):
                    return False
    return True


def are_SparseNdarrays_equal(x, y):
    if x._shape != y._shape:
        return False
    contents1 = x._contents
    contents2 = y._contents

    if isinstance(contents1, list):
        if isinstance(contents2, list):
            ndim = len(x._shape)
            return _recursive_compare_contents(contents1, contents2, dim=ndim - 1)
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
    assert repr(y).find("SparseNdarray") > 0

    with pytest.raises(ValueError, match="match the extent"):
        y = delayedarray.SparseNdarray((10, 15, 1), contents)

    with pytest.raises(ValueError, match="out of range"):
        y = delayedarray.SparseNdarray((5, 15, 20), contents)

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

    with pytest.raises(ValueError, match="inconsistent data type"):
        y = delayedarray.SparseNdarray(test_shape, contents, dtype=numpy.int32)

    with pytest.raises(ValueError, match="cannot infer 'dtype'"):
        y = delayedarray.SparseNdarray(test_shape, None)

    empty = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.int32, index_dtype=numpy.int32)
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
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(10, 30), slice(None), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
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
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(20, 30), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert (sliced == output[(..., *slices)]).all()

    slices = (slice(None), slice(10, 80))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
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
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
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
    slices = (slice(5, 48, 5), slice(0, 30, 3))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(20, 30), slice(None))
    sliced = y[slices]
    assert (convert_SparseNdarray_to_numpy(sliced) == ref[(..., *slices)]).all()

    slices = (slice(None), slice(10, 25))
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
    dout = delayedarray.extract_dense_array(y, full_indices)
    assert dout.dtype == numpy.int16
    ref = convert_SparseNdarray_to_numpy(y)
    assert (dout == ref).all()

    spout = delayedarray.extract_sparse_array(y, full_indices)
    assert spout.dtype == numpy.int16
    assert (convert_SparseNdarray_to_numpy(spout) == ref).all()


def test_SparseNdarray_empty():
    test_shape = (20, 21, 22)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.uint32, index_dtype=numpy.int32)
    assert y.shape == test_shape
    assert y.dtype == numpy.uint32

    full_indices = [range(d) for d in test_shape]
    dout = delayedarray.extract_dense_array(y, full_indices)
    assert (dout == numpy.zeros(test_shape)).all()
    dout = delayedarray.extract_dense_array(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert (dout == numpy.zeros((3, 4, 5))).all()

    spout = delayedarray.extract_sparse_array(y, full_indices)
    assert spout._contents is None
    assert spout.shape == test_shape
    assert spout.dtype == numpy.uint32
    spout = delayedarray.extract_sparse_array(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert spout.shape == (3, 4, 5)


#######################################################
#######################################################


def test_SparseNdarray_subset_simple():
    test_shape = (20, 21, 22)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = convert_SparseNdarray_to_numpy(y)

    # No-op subset.
    subset = (slice(None), slice(None))
    sub = y[subset]
    assert (numpy.array(sub) == ref[subset]).all()

    # Consecutive subset.
    subset = (slice(2, 18), slice(3, 20), slice(5, 22))
    sub = y[subset]
    assert (numpy.array(sub) == ref[subset]).all()

    # Increasing non-consecutive subset.
    subset = (slice(2, 18, 2), slice(3, 20, 2), slice(1, 22, 2))
    sub = y[subset]
    assert (numpy.array(sub) == ref[subset]).all()

    # Unsorted subset.
    subset = [list(range(s)) for s in test_shape]
    for s in subset:
        numpy.random.shuffle(s)
    sub = y[numpy.ix_(*subset)]
    assert (numpy.array(sub) == ref[numpy.ix_(*subset)]).all()

    # Duplicated subset.
    subset = []
    for s in test_shape:
        cursub = []
        for i in range(s):
            cursub += [i] * numpy.random.randint(4)
        subset.append(cursub)

    sub = y[numpy.ix_(*subset)]
    assert (numpy.array(sub) == ref[numpy.ix_(*subset)]).all()


def test_SparseNdarray_subset_collapse():
    test_shape = (20, 50)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = convert_SparseNdarray_to_numpy(y)

    first = y[0,:]
    assert isinstance(first, numpy.ndarray)
    assert (first == ref[0,:]).all()

    first = y[:,1]
    assert isinstance(first, numpy.ndarray)
    assert (first == ref[:,1]).all()


#######################################################
#######################################################


def test_SparseNdarray_abs():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    out = abs(y)
    assert (numpy.array(out) == abs(numpy.array(y))).all()

    # Checking that the transformer does something sensible here.
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.float64, index_dtype=numpy.int32)
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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y + y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref + ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y - y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref - ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y * y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref * ref2)).all()


def _equal_with_nan(left, right):
    missing = numpy.isnan(left)
    assert (missing == numpy.isnan(right)).all()
    left[missing] = 0
    right[missing] = 0
    assert (left == right).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y / y2 
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, ref / ref2)


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.float64)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y // y2 
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, ref // ref2)


def test_SparseNdarray_modulo():
    test_shape = (30, 40)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 % y
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, 1.5 % ref)
    out = y % 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % 2).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % other).all()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref % other).all()

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.float64)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y % y2 
        assert isinstance(out, numpy.ndarray)
        _equal_with_nan(out, ref % ref2)


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=1, upper=5, dtype=numpy.float64)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y ** y2 
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref ** ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y == y2 
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref == ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y != y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref != ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y >= y2 
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref >= ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y > y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref > ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y <= y2 
    assert isinstance(out, numpy.ndarray)
    assert (out == (ref <= ref2)).all()


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

    contents2 = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape, contents2)
    ref2 = numpy.array(y2)
    out = y < y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == (ref < ref2)).all()


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


#######################################################
#######################################################


def test_SparseNdarray_transpose():
    test_shape = (50, 30, 20)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    out = numpy.transpose(y, axes=[1, 2, 0])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == numpy.transpose(ref, axes=[1, 2, 0])).all()

    out = numpy.transpose(y, axes=[0, 2, 1])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == numpy.transpose(ref, axes=[0, 2, 1])).all()

    out = numpy.transpose(y, axes=[1, 0, 2])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == numpy.transpose(ref, axes=[1, 0, 2])).all()

    out = y.T
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref.T).all()

    # No-op for 1-dimensional arrays.
    test_shape = (50,)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)
    out = numpy.transpose(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == ref).all()

    # Works for Nones.
    test_shape = (20, 30)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.float64, index_dtype=numpy.int32)
    ref = numpy.zeros(test_shape)
    out = numpy.transpose(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == numpy.transpose(ref)).all()


#######################################################
#######################################################


def test_SparseNdarray_concatenate_3d():
    test_shape = (10, 20, 30)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    # Combining on the first dimension.
    test_shape2 = (5, 20, 30)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2))).all()

    # Combining on the middle dimension.
    test_shape2 = (10, 15, 30)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2), axis=1)).all()

    # Combining on the last dimension.
    test_shape2 = (10, 20, 15)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2), axis=2)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2), axis=2)).all()


def test_SparseNdarray_concatenate_2d():
    test_shape = (55, 20)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    # Combining on the first dimension.
    test_shape2 = (25, 20)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2))).all()

    # Combining on the last dimension.
    test_shape2 = (55, 15)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2), axis=1)).all()


def test_SparseNdarray_concatenate_1d():
    test_shape = (10,)
    contents = mock_SparseNdarray_contents(test_shape, lower=-100, upper=100, dtype=numpy.int16)
    y = delayedarray.SparseNdarray(test_shape, contents)
    ref = numpy.array(y)

    test_shape2 = (5,)
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2))).all()

    # One dimension plus None's.
    test_shape2 = (5,)
    y2 = delayedarray.SparseNdarray(test_shape2, None, dtype=numpy.float64, index_dtype=numpy.int32)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2))).all()


def test_SparseNdarray_concatenate_nones():
    test_shape = (10, 20)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.float64, index_dtype=numpy.int32)
    ref = numpy.array(y)

    test_shape2 = (10, 25)
    y2 = delayedarray.SparseNdarray(test_shape2, None, dtype=numpy.float64, index_dtype=numpy.int32)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2), axis=1)).all()

    # Partial none.
    contents2 = mock_SparseNdarray_contents(test_shape2, lower=-100, upper=100, dtype=numpy.int16)
    y2 = delayedarray.SparseNdarray(test_shape2, contents2)
    ref2 = numpy.array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert (numpy.array(combined) == numpy.concatenate((ref, ref2), axis=1)).all()
