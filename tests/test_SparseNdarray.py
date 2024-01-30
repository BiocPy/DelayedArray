import copy
import warnings
import random
import delayedarray
import numpy
import pytest

from utils import assert_identical_ndarrays, assert_close_ndarrays, safe_concatenate, mock_SparseNdarray_contents, simulate_SparseNdarray

#######################################################
#######################################################


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
    if x.is_masked:
        output = numpy.ma.MaskedArray(output)
    for pos, val in triplets:
        output[pos] = val

    return output


def _compare_sparse_vectors(left, right):
    idx_l, val_l = left
    idx_r, val_r = right

    assert len(idx_l) == len(idx_r)
    assert (idx_l == idx_r).all()
    comp = (val_l == val_r)

    masked = numpy.ma.isMaskedArray(val_l) 
    assert masked == numpy.ma.isMaskedArray(val_r) 
    if masked:
        assert (val_l.mask == val_r.mask).all()
        assert numpy.logical_or(comp, val_r.mask).all()
    else:
        assert comp.all()


def _recursive_compare_contents(left, right, dim):
    assert len(left) == len(right)

    if dim == 1:
        for i, lcon in enumerate(left):
            is_none = (lcon is None)
            assert is_none == (right[i] is None)
            if not is_none:
                _compare_sparse_vectors(lcon, right[i])
    else:
        for i, lcon in enumerate(left):
            is_none = (lcon is None)
            assert is_none == (right[i] is None)
            if not is_none:
                _recursive_compare_contents(lcon, right[i], dim - 1)


def assert_identical_SparseNdarrays(x, y):
    assert x._shape == y._shape
    contents1 = x._contents
    contents2 = y._contents

    is_list = isinstance(contents1, list)
    assert is_list == isinstance(contents2, list)

    if is_list:
        ndim = len(x._shape)
        _recursive_compare_contents(contents1, contents2, dim=ndim - 1)
    else:
        is_none = contents1 is None
        assert is_none == (contents2 is None)
        if not is_none:
            _compare_sparse_vectors(contents1, contents2)


def slices2ranges(slices, shape):
    output = []
    for i, s in enumerate(slices):
        output.append(range(*s.indices(shape[i])))
    return (*output,)

#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_check(mask_rate):
    test_shape = (10, 15, 20)
    contents = mock_SparseNdarray_contents(test_shape, mask_rate=mask_rate)
    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.shape == test_shape
    assert y.dtype == numpy.float64
    assert repr(y).find("SparseNdarray") > 0
    assert delayedarray.is_sparse(y)
    assert delayedarray.is_masked(y) == (mask_rate > 0)
    assert delayedarray.is_masked(y) == y.is_masked

    clone = copy.copy(y)
    clone.contents[0] = "FOOBAR"
    assert y.contents[0] != "FOOBAR"

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
        y = delayedarray.SparseNdarray(test_shape, contents, dtype=numpy.dtype("int32"))

    with pytest.raises(ValueError, match="cannot infer 'dtype'"):
        y = delayedarray.SparseNdarray(test_shape, None)

    empty = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.dtype("int32"), index_dtype=numpy.dtype("int32"))
    assert empty.shape == test_shape
    assert empty.dtype == numpy.int32
    assert not empty.is_masked


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_dense_array_3d(mask_rate):
    test_shape = (16, 32, 8)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)

    # Full extraction.
    output = delayedarray.to_dense_array(y)
    assert_identical_ndarrays(output, convert_SparseNdarray_to_numpy(y))

    # Sliced extraction.
    slices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])

    slices = (slice(None), slice(0, 20, 2), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])

    slices = (slice(None), slice(None), slice(0, 8, 2))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])

    slices = (slice(10, 30), slice(None), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_dense_array_2d(mask_rate):
    test_shape = (50, 100)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)

    # Full extraction.
    output = delayedarray.to_dense_array(y)
    assert_identical_ndarrays(output, convert_SparseNdarray_to_numpy(y))

    # Sliced extraction.
    slices = (slice(5, 48, 5), slice(0, 90, 3))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])

    slices = (slice(20, 30), slice(None))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])

    slices = (slice(None), slice(10, 80))
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_dense_array_1d(mask_rate):
    test_shape = (99,)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)
    assert y.dtype == numpy.float64

    # Full extraction.
    output = delayedarray.to_dense_array(y)
    assert_identical_ndarrays(output, convert_SparseNdarray_to_numpy(y))

    # Sliced extraction.
    slices = (slice(5, 90, 7),)
    sliced = delayedarray.extract_dense_array(y, slices2ranges(slices, test_shape))
    assert_identical_ndarrays(sliced, output[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_sparse_array_3d(mask_rate):
    test_shape = (20, 15, 10)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)

    # Full extraction.
    full = [slice(None)] * len(test_shape)
    output = y[(*full,)]
    assert_identical_SparseNdarrays(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(2, 15, 3), slice(0, 20, 2), slice(4, 8))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])

    slices = (slice(test_shape[0]), slice(0, 20, 2), slice(test_shape[2]))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])

    slices = (slice(test_shape[0]), slice(test_shape[1]), slice(0, 8, 2))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])

    slices = (slice(10, 30), slice(test_shape[1]), slice(test_shape[2]))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_sparse_array_2d(mask_rate):
    test_shape = (99, 40)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)

    # Full extraction.
    full = [slice(None)] * len(test_shape)
    output = y[(*full,)]
    assert_identical_SparseNdarrays(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(5, 48, 5), slice(0, 30, 3))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])

    slices = (slice(20, 30), slice(None))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])

    slices = (slice(None), slice(10, 25))
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_extract_sparse_array_1d(mask_rate):
    test_shape = (99,)
    y = simulate_SparseNdarray(test_shape, mask_rate=mask_rate)

    # Full extraction.
    full = (slice(None),)
    output = y[(*full,)]
    assert_identical_SparseNdarrays(output, y)

    ref = convert_SparseNdarray_to_numpy(y)

    # Sliced extraction.
    slices = (slice(5, 90, 7),)
    sliced = y[slices]
    assert_identical_ndarrays(convert_SparseNdarray_to_numpy(sliced), ref[slices])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_int_type(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    assert y.shape == test_shape
    assert y.dtype == numpy.int16

    full_indices = [range(d) for d in test_shape]
    dout = delayedarray.extract_dense_array(y, full_indices)
    assert dout.dtype == numpy.int16
    ref = convert_SparseNdarray_to_numpy(y)
    assert_identical_ndarrays(dout, ref)

    spout = delayedarray.extract_sparse_array(y, full_indices)
    assert spout.dtype == numpy.int16
    assert_identical_ndarrays(dout, ref)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_empty(mask_rate):
    test_shape = (20, 21, 22)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.uint32, index_dtype=numpy.dtype("int32"))
    assert y.shape == test_shape
    assert y.dtype == numpy.uint32

    full_indices = [range(d) for d in test_shape]
    dout = delayedarray.extract_dense_array(y, full_indices)
    assert_identical_ndarrays(dout, numpy.zeros(test_shape))
    dout = delayedarray.extract_dense_array(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert_identical_ndarrays(dout, numpy.zeros((3, 4, 5)))

    spout = delayedarray.extract_sparse_array(y, full_indices)
    assert spout._contents is None
    assert spout.shape == test_shape
    assert spout.dtype == numpy.uint32
    spout = delayedarray.extract_sparse_array(y, ([1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]))
    assert spout.shape == (3, 4, 5)


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_subset_simple(mask_rate):
    test_shape = (20, 21, 22)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = convert_SparseNdarray_to_numpy(y)

    # No-op subset.
    subset = (slice(None), slice(None))
    sub = y[subset]
    assert_identical_SparseNdarrays(sub, y)

    # Consecutive subset.
    subset = (slice(2, 18), slice(3, 20), slice(5, 22))
    sub = y[subset]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), ref[subset])

    # Increasing non-consecutive subset.
    subset = (slice(2, 18, 2), slice(3, 20, 2), slice(1, 22, 2))
    sub = y[subset]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), ref[subset])

    # Unsorted subset.
    subset = [list(range(s)) for s in test_shape]
    for s in subset:
        numpy.random.shuffle(s)
    sub = y[numpy.ix_(*subset)]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), ref[numpy.ix_(*subset)])

    # Duplicated subset.
    subset = []
    for s in test_shape:
        cursub = []
        for i in range(s):
            cursub += [i] * numpy.random.randint(4)
        subset.append(cursub)
    sub = y[numpy.ix_(*subset)]
    assert_identical_ndarrays(delayedarray.to_dense_array(sub), ref[numpy.ix_(*subset)])


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_subset_collapse(mask_rate):
    test_shape = (20, 50)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = convert_SparseNdarray_to_numpy(y)

    first = y[0,:]
    assert isinstance(first, numpy.ndarray)
    assert_identical_ndarrays(first, ref[0,:])

    first = y[:,1]
    assert isinstance(first, numpy.ndarray)
    assert_identical_ndarrays(first, ref[:,1])


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_abs(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    out = abs(y)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), abs(delayedarray.to_dense_array(y)))

    # Checking that the transformer does something sensible here.
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int32"))
    out = abs(y)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.zeros(test_shape))

    test_shape = (99,)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"))
    out = abs(y)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), abs(delayedarray.to_dense_array(y)))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_neg(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    assert_identical_ndarrays(delayedarray.to_dense_array(-y), -delayedarray.to_dense_array(y))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_ufunc_simple(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = numpy.log1p(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert out.dtype == numpy.float32
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.log1p(ref))

    out = numpy.exp(y)
    assert isinstance(out, numpy.ndarray)
    assert out.dtype == numpy.float32
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.exp(ref))


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_add(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1 + y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1 + ref)
    out = y + 2
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref + 2)

    other = numpy.random.rand(40)
    out = other + y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other + ref)
    out = y + other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref + other)

    other = numpy.random.rand(30, 1)
    out = other + y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other + ref)
    out = y + other 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref + other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y + y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref + ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_sub(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 - y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1.5 - ref)
    out = y - 2.5
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref - 2.5)

    other = numpy.random.rand(40)
    out = other - y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other - ref)
    out = y - other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref - other)

    other = numpy.random.rand(30, 1)
    out = other - y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other - ref)
    out = y - other 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref - other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"))
    ref2 = delayedarray.to_dense_array(y2)
    out = y - y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref - ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_multiply(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), 1.5 * ref)
    out = y * 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref * 2)

    other = numpy.random.rand(40)
    out = other * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other * ref)
    out = y * other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref * other)

    other = numpy.random.rand(30, 1)
    out = other * y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other * ref)
    out = y * other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref * other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y * y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref * ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_divide(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 / y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, 1.5 / ref)
    out = y / 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref / 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other / y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other / ref)
    out = y / other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref / other)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other / y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other / ref)
    out = y / other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref / other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y / y2 
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, ref / ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_floor_divide(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 // y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, 1.5 // ref)
    out = y // 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref // 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other // y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other // ref)
    out = y // other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref // other)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other // y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other // ref)
    out = y // other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref // other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y // y2 
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, ref // ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_modulo(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 1.5 % y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, 1.5 % ref)
    out = y % 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref % 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(40)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref % other)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        other = numpy.random.rand(30, 1)
        out = other % y
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, other % ref)
    out = y % other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref % other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = y % y2 
        assert isinstance(out, numpy.ndarray)
        assert_identical_ndarrays(out, ref % ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_power(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 ** y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1.5 ** ref)
    out = y ** 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref ** 2)

    other = numpy.random.rand(40)
    out = other ** y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other ** ref)
    out = y ** other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref ** other)

    other = numpy.random.rand(30, 1)
    out = other ** y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other ** ref)
    out = y ** other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref ** other)

    y2 = simulate_SparseNdarray(test_shape, lower=1, upper=5, dtype=numpy.dtype("float64"))
    ref2 = delayedarray.to_dense_array(y2)
    out = y ** y2 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, (ref ** ref2))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_equal(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), 1.5 == ref)
    out = y == 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref == 2)

    other = numpy.random.rand(40)
    out = other == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other == ref)
    out = y == other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref == other)

    other = numpy.random.rand(30, 1)
    out = other == y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other == ref)
    out = y == other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref == other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y == y2 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref == ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_not_equal(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 != y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1.5 != ref)
    out = y != 2
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref != 2)

    other = numpy.random.rand(40)
    out = other != y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other != ref)
    out = y != other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref != other)

    other = numpy.random.rand(30, 1)
    out = other != y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other != ref)
    out = y != other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref != other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y != y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref != ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_greater_than_or_equal(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 >= y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1.5 >= ref)
    out = y >= 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref >= 2)

    other = numpy.random.rand(40)
    out = other >= y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other >= ref)
    out = y >= other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref >= other)

    other = numpy.random.rand(30, 1)
    out = other >= y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other >= ref)
    out = y >= other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref >= other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y >= y2 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref >= ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_greater(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 > y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, 1.5 > ref)
    out = y > 2
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref > 2)

    other = numpy.random.rand(40)
    out = other > y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other > ref)
    out = y > other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref > other)

    other = numpy.random.rand(30, 1)
    out = other > y
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, other > ref)
    out = y > other
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref > other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y > y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref > ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_less_than_or_equal(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), 1.5 <= ref)
    out = y <= 2
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref <= 2)

    other = numpy.random.rand(40)
    out = other <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other <= ref)
    out = y <= other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref <= other)

    other = numpy.random.rand(30, 1)
    out = other <= y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other <= ref)
    out = y <= other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref <= other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y <= y2 
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref <= ref2)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_less_than_or_equal(mask_rate):
    test_shape = (30, 40)
    y = simulate_SparseNdarray(test_shape, lower=1, upper=10, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = 1.5 < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), 1.5 < ref)
    out = y < 2
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref < 2)

    other = numpy.random.rand(40)
    out = other < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other < ref)
    out = y < other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref < other)

    other = numpy.random.rand(30, 1)
    out = other < y
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), other < ref)
    out = y < other
    assert isinstance(out, numpy.ndarray)
    assert_identical_ndarrays(out, ref < other)

    y2 = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)
    out = y < y2 
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref < ref2)


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_astype(mask_rate):
    test_shape = (50, 30, 20)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)

    z = y.astype(numpy.float64)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert_identical_SparseNdarrays(z, y)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_round(mask_rate):
    test_shape = (50, 30, 20)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("float64"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    z = numpy.round(y)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.round(ref))

    z = numpy.round(y, decimals=1)
    assert isinstance(z, delayedarray.SparseNdarray)
    assert z.dtype == numpy.float64
    assert_identical_ndarrays(delayedarray.to_dense_array(z), numpy.round(ref, decimals=1))


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_transpose(mask_rate):
    test_shape = (50, 30, 20)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    out = numpy.transpose(y, axes=[1, 2, 0])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.transpose(ref, axes=[1, 2, 0]))

    out = numpy.transpose(y, axes=[0, 2, 1])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.transpose(ref, axes=[0, 2, 1]))

    out = numpy.transpose(y, axes=[1, 0, 2])
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.transpose(ref, axes=[1, 0, 2]))

    out = y.T
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref.T)

    # No-op for 1-dimensional arrays.
    test_shape = (50,)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)
    out = numpy.transpose(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), ref)

    # Works for Nones.
    test_shape = (20, 30)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int32"))
    ref = numpy.zeros(test_shape)
    out = numpy.transpose(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(out), numpy.transpose(ref))


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_concatenate_3d(mask_rate):
    test_shape = (10, 20, 30)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    # Combining on the first dimension.
    test_shape2 = (5, 20, 30)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2)))

    # Combining on the middle dimension.
    test_shape2 = (10, 15, 30)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2), axis=1))

    # Combining on the last dimension.
    test_shape2 = (10, 20, 15)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2), axis=2)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2), axis=2))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_concatenate_2d(mask_rate):
    test_shape = (55, 20)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    # Combining on the first dimension.
    test_shape2 = (25, 20)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2)))

    # Combining on the last dimension.
    test_shape2 = (55, 15)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2), axis=1))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_concatenate_1d(mask_rate):
    test_shape = (10,)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    test_shape2 = (5,)
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2)))

    # One dimension plus None's.
    test_shape2 = (5,)
    y2 = delayedarray.SparseNdarray(test_shape2, None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int32"))
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2))
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2)))


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_concatenate_nones(mask_rate):
    test_shape = (10, 20)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int32"))
    ref = delayedarray.to_dense_array(y)

    test_shape2 = (10, 25)
    y2 = delayedarray.SparseNdarray(test_shape2, None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int32"))
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2), axis=1))

    # Partial none.
    y2 = simulate_SparseNdarray(test_shape2, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref2 = delayedarray.to_dense_array(y2)

    combined = numpy.concatenate((y, y2), axis=1)
    assert isinstance(combined, delayedarray.SparseNdarray)
    assert_identical_ndarrays(delayedarray.to_dense_array(combined), safe_concatenate((ref, ref2), axis=1))


#######################################################
#######################################################


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_sum(mask_rate):
    test_shape = (10, 20, 30)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    assert ref.sum() == y.sum()
    assert_identical_ndarrays(ref.sum(axis=1), y.sum(axis=1))
    assert_identical_ndarrays(ref.sum(axis=-1), y.sum(axis=-1))
    assert_identical_ndarrays(ref.sum(axis=(0, 2)), y.sum(axis=(0, 2)))

    # Trying with a single dimension.
    test_shape = (100,)
    y = numpy.round(simulate_SparseNdarray(test_shape, lower=-5, upper=5, mask_rate=mask_rate))
    ref = delayedarray.to_dense_array(y)
    assert ref.sum() == y.sum()

    # Checking that full masking is respected.
    y = delayedarray.SparseNdarray((1,), (numpy.array([0]), numpy.ma.MaskedArray([1], mask=True)))
    assert y.sum() is numpy.ma.masked


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_mean(mask_rate):
    test_shape = (10, 20, 30)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    assert numpy.isclose(ref.mean(), y.mean())
    assert_close_ndarrays(ref.mean(axis=1), y.mean(axis=1))
    assert_close_ndarrays(ref.mean(axis=-1), y.mean(axis=-1))
    assert_close_ndarrays(ref.mean(axis=(0, 2)), y.mean(axis=(0, 2)))

    # Trying with a single dimension.
    test_shape = (100,)
    y = numpy.round(simulate_SparseNdarray(test_shape, lower=-5, upper=5, mask_rate=mask_rate))
    ref = delayedarray.to_dense_array(y)
    assert numpy.isclose(ref.mean(), y.mean())

    # Checking that full masking is respected.
    y = delayedarray.SparseNdarray((1,), (numpy.array([0]), numpy.ma.MaskedArray([1], mask=True)))
    assert y.mean() is numpy.ma.masked

    # Checking that an empty vector behaves correctly.
    y = delayedarray.SparseNdarray((0,), None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int8"))
    with pytest.warns(RuntimeWarning):
        assert numpy.isnan(y.mean())


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_SparseNdarray_var(mask_rate):
    test_shape = (10, 20, 30)
    y = simulate_SparseNdarray(test_shape, lower=-100, upper=100, dtype=numpy.dtype("int16"), mask_rate=mask_rate)
    ref = delayedarray.to_dense_array(y)

    assert numpy.isclose(ref.var(), y.var())
    assert_close_ndarrays(ref.var(axis=1), y.var(axis=1))
    assert_close_ndarrays(ref.var(axis=-1), y.var(axis=-1))
    assert_close_ndarrays(ref.var(axis=(0, 2)), y.var(axis=(0, 2)))

    # Trying with a single dimension.
    test_shape = (100,)
    y = numpy.round(simulate_SparseNdarray(test_shape, lower=-5, upper=5, mask_rate=mask_rate))
    ref = delayedarray.to_dense_array(y)
    assert numpy.isclose(ref.var(), y.var())

    # Checking that full masking is respected.
    y = delayedarray.SparseNdarray((1,), (numpy.array([0]), numpy.ma.MaskedArray([1], mask=True)))
    with pytest.warns(RuntimeWarning):
        assert y.var() is numpy.ma.masked

    # Checking that an empty vector behaves correctly.
    y = delayedarray.SparseNdarray((0,), None, dtype=numpy.dtype("float64"), index_dtype=numpy.dtype("int8"))
    with pytest.warns(RuntimeWarning):
        assert numpy.isnan(y.var())
