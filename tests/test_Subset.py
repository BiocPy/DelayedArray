
import delayedarray
from utils import *
import numpy
import random


def test_Subset_consecutive_dense():
    test_shape = (20, 30)
    y = numpy.random.rand(*test_shape)

    subset = (range(20), range(30))
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y).all()

    # Sliced:
    slices = ([1, 3, 5, 7, 9], [2, 4, 6, 8])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_sorted_unique_dense():
    test_shape = (10, 20, 30)
    y = numpy.random.rand(*test_shape)

    subset = (range(2, 7), range(0, 20, 2), range(5, 25, 3))
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(
        sub, (slice(None), slice(None), slice(None))
    )
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = ([0, 2, 4], [2, 4, 6, 8], [0, 1, 2, 3, 5])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_sorted_dense():
    test_shape = (50, 10)
    y = numpy.random.rand(*test_shape)

    subset = (
        [5, 5, 5, 10, 20, 20, 25, 30, 30, 30, 40, 45, 45, 45],
        [0, 0, 0, 2, 5, 5, 5, 6, 8, 8, 8, 8, 8, 9, 9],
    )
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = ([0, 2, 4, 6, 8, 10], [3, 6, 9, 12])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_unique_dense():
    test_shape = (20, 10, 30)
    y = numpy.random.rand(*test_shape)

    subset = []
    for i in range(len(test_shape)):
        current = list(range(i, test_shape[i], i + 1))
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(
        sub, (slice(None), slice(None), slice(None))
    )
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = (
        range(3, len(subset[0]), 2),
        range(0, len(subset[1]), 2),
        range(1, len(subset[2]), 2),
    )
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_general_dense():
    test_shape = (40, 30)
    y = numpy.random.rand(*test_shape)

    subset = []
    for i in range(len(test_shape)):
        current = []
        for x in range(i, test_shape[i], 5):
            current += [x] * int(random.uniform(0, 1) * 5)
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (full == y[numpy.ix_(*subset)]).all()

    # Sliced:
    slices = (range(0, len(subset[0]), 2), range(3, len(subset[1]), 2))
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (full[numpy.ix_(*slices)] == partial).all()


def test_Subset_1d_dense():
    y = numpy.random.rand(100)

    subset = ([10, 20, 30, 40, 70, 80, 90],)
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)

    full = delayedarray.extract_dense_array(sub, (slice(None),))
    assert (full == y[subset[0]]).all()

    slices = ([2, 4, 6],)
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (partial == full[slices[0]]).all()


def test_Subset_consecutive_sparse():
    test_shape = (21, 17, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (range(21), range(17), range(30))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == test_shape

    # Full:
    full = delayedarray.extract_sparse_array(
        sub, (slice(None), slice(None), slice(None))
    )
    assert are_SparseNdarrays_equal(full, y)

    # Sliced:
    slices = ([1, 3, 5, 7, 9], [2, 4, 6, 8], [10, 12, 14, 16])
    partial = delayedarray.extract_sparse_array(sub, slices)
    ref = delayedarray.extract_sparse_array(y, slices)
    assert are_SparseNdarrays_equal(partial, ref)


def test_Subset_sorted_unique_sparse():
    test_shape = (50, 30)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (range(10, 40, 3), range(1, 28, 2))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full = delayedarray.extract_sparse_array(sub, (slice(None), slice(None)))
    ref = delayedarray.extract_sparse_array(y, subset)
    assert are_SparseNdarrays_equal(full, ref)

    # Sliced:
    slices = ([0, 2, 4, 6, 8], [1, 3, 7, 9, 11, 13])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = []
    for i in range(2):
        current = []
        for s in slices[i]:
            current.append(subset[i][s])
        refsub.append(current)
    ref = delayedarray.extract_sparse_array(y, (*refsub,))
    assert are_SparseNdarrays_equal(partial, ref)


def test_Subset_sorted_sparse():
    test_shape = (30, 20)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = (
        [3, 6, 6, 9, 12, 12, 12, 15, 21, 24, 24, 24, 24],
        [5, 6, 7, 7, 8, 14, 14, 15, 16, 16, 18],
    )
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    ref = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == ref).all()

    # Sliced:
    slices = ([2, 5, 8, 11], [2, 3, 4, 7, 8, 9])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = ref[numpy.ix_(*slices)]
    assert (delayedarray.extract_dense_array(partial, full_indices) == refsub).all()


def test_Subset_unique_sparse():
    test_shape = (20, 15, 25)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = []
    for i in range(len(test_shape)):
        current = list(range(i, test_shape[i], len(test_shape) - i))
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    ref = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == ref).all()

    # Sliced:
    slices = ([2, 4, 6], [1, 3, 5], [4, 5, 10, 15, 20, 21])
    partial = delayedarray.extract_sparse_array(sub, slices)
    refsub = ref[numpy.ix_(*slices)]
    assert (delayedarray.extract_dense_array(partial, full_indices) == refsub).all()


def test_Subset_general_sparse():
    test_shape = (25, 55)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)

    subset = []
    for i in range(len(test_shape)):
        current = []
        for x in range(i, test_shape[i], 5):
            current += [x] * int(random.uniform(0, 1) * 5)
        random.shuffle(current)
        subset.append(current)

    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)

    # Full:
    full_indices = (slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    refsub = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == refsub).all()

    # Sliced:
    slices = (range(2, len(subset[0]), 2), range(1, len(subset[1]), 2))
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        delayedarray.extract_dense_array(partial, full_indices)
        == refsub[numpy.ix_(*slices)]
    ).all()


def test_Subset_1d_sparse():
    contents = mock_SparseNdarray_contents((100,))
    y = delayedarray.SparseNdarray((100,), contents)

    subset = ([10, 20, 30, 40, 70, 80, 90],)
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)

    full = delayedarray.extract_sparse_array(sub, (slice(None),))
    ref = delayedarray.extract_dense_array(y, (slice(None),))[subset[0]]
    assert (delayedarray.extract_dense_array(full, (slice(None),)) == ref).all()

    slices = ([2, 4, 6],)
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        delayedarray.extract_dense_array(partial, (slice(None),)) == ref[slices[0]]
    ).all()


def test_Subset_empty():
    test_shape = (100, 200, 300)
    y = delayedarray.SparseNdarray(test_shape, None, dtype=numpy.float32)

    subset = (range(10, 90, 10), range(5, 180, 5), range(100, 250, 10))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (*[len(x) for x in subset],)
    assert sub.dtype == numpy.float32

    full_indices = (slice(None), slice(None), slice(None))
    full = delayedarray.extract_sparse_array(sub, full_indices)
    ref = delayedarray.extract_dense_array(y, full_indices)[numpy.ix_(*subset)]
    assert (delayedarray.extract_dense_array(full, full_indices) == ref).all()

    slices = ([0, 2, 4, 6], [0, 1, 2, 3, 4, 10, 20], [1, 3, 5, 7, 9, 11])
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        delayedarray.extract_dense_array(partial, full_indices)
        == ref[numpy.ix_(*slices)]
    ).all()


def test_Subset_dimension_lost_dense():
    test_shape = (10, 20, 30)
    y = numpy.random.rand(*test_shape)

    # First dimension lost.
    subset = (5, range(0, 20, 2), range(0, 30, 2))
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (10, 15)
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (ext == y[5, :, :][numpy.ix_(*subset[1:])]).all()

    slices = (range(0, 5), range(10, 15))
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (ext[numpy.ix_(*slices)] == partial).all()

    # Last dimension lost.
    subset = (range(0, 10), range(2, 20, 3), 20)
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (10, 6)
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_dense_array(sub, (slice(None), slice(None)))
    assert (ext == y[:, :, 20][numpy.ix_(*subset[:-1])]).all()

    slices = (range(0, 5), [1, 3, 5])
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (ext[numpy.ix_(*slices)] == partial).all()

    # Multiple dimensions lost.
    subset = (5, [5, 8, 11, 14, 17], 20)
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == (5,)
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_dense_array(sub, (slice(None),))
    assert (ext == y[5, :, 20][subset[1]]).all()

    slices = ([0, 2, 4],)
    partial = delayedarray.extract_dense_array(sub, slices)
    assert (ext[numpy.ix_(*slices)] == partial).all()

    # All dimensions lost.
    subset = (5, 10, 20)
    sub = delayedarray.Subset(y, subset)
    assert not delayedarray.is_sparse(sub)
    assert sub.shape == ()
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_dense_array(sub, ())
    assert ext.shape == ()
    assert (ext == numpy.array(y[5, 10, 20])).all()


def test_Subset_dimension_lost_sparse():
    test_shape = (40, 30, 50)
    contents = mock_SparseNdarray_contents(test_shape)
    y = delayedarray.SparseNdarray(test_shape, contents)
    rawref = delayedarray.extract_dense_array(y, (*([slice(None)] * 3),))

    # First dimension lost.
    subset = (5, range(5, 20, 2), range(10, 40, 2))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (8, 15)
    assert sub.dtype == numpy.float64

    full_indices = (slice(None), slice(None))
    ext = delayedarray.extract_sparse_array(sub, full_indices)
    ref = rawref[5, :, :][numpy.ix_(*subset[1:])]
    assert (delayedarray.extract_dense_array(ext, full_indices) == ref).all()

    slices = (range(2, 6), range(10, 15))
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        ref[numpy.ix_(*slices)]
        == delayedarray.extract_dense_array(partial, full_indices)
    ).all()

    # Last dimension lost.
    subset = (range(22, 38), range(2, 25, 3), 20)
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (16, 8)
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_sparse_array(sub, full_indices)
    ref = rawref[:, :, 20][numpy.ix_(*subset[:-1])]
    assert (delayedarray.extract_dense_array(ext, full_indices) == ref).all()

    slices = (range(4, 12), [2, 4, 6])
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        ref[numpy.ix_(*slices)]
        == delayedarray.extract_dense_array(partial, full_indices)
    ).all()

    # Multiple dimensions lost.
    subset = (7, range(4, 28, 3), 19)
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (8,)
    assert sub.dtype == numpy.float64

    full_indices = (slice(None),)
    ext = delayedarray.extract_sparse_array(sub, full_indices)
    ref = rawref[7, :, 19][subset[1]]
    assert (delayedarray.extract_dense_array(ext, full_indices) == ref).all()

    slices = ([1, 3, 5, 7],)
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        ref[numpy.ix_(*slices)]
        == delayedarray.extract_dense_array(partial, full_indices)
    ).all()

    # Multiple dimensions lost (II).
    subset = (range(0, 40, 5), 20, 19)
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (8,)
    assert sub.dtype == numpy.float64

    full_indices = (slice(None),)
    ext = delayedarray.extract_sparse_array(sub, full_indices)
    ref = rawref[:, 20, 19][subset[0]]
    assert (delayedarray.extract_dense_array(ext, full_indices) == ref).all()

    slices = ([1, 3, 5, 7],)
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        ref[numpy.ix_(*slices)]
        == delayedarray.extract_dense_array(partial, full_indices)
    ).all()

    # Multiple dimensions lost (III).
    subset = (5, 10, range(1, 50, 7))
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == (7,)
    assert sub.dtype == numpy.float64

    full_indices = (slice(None),)
    ext = delayedarray.extract_sparse_array(sub, full_indices)
    ref = rawref[5, 10, :][subset[2]]
    assert (delayedarray.extract_dense_array(ext, full_indices) == ref).all()

    slices = ([0, 2, 4, 6],)
    partial = delayedarray.extract_sparse_array(sub, slices)
    assert (
        ref[numpy.ix_(*slices)]
        == delayedarray.extract_dense_array(partial, full_indices)
    ).all()

    # All dimensions lost.
    subset = (5, 10, 20)
    sub = delayedarray.Subset(y, subset)
    assert delayedarray.is_sparse(sub)
    assert sub.shape == ()
    assert sub.dtype == numpy.float64

    ext = delayedarray.extract_sparse_array(sub, ())
    assert ext.shape == ()
