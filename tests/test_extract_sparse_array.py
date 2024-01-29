import delayedarray
import numpy
import scipy.sparse


def test_extract_sparse_array_csc_matrix():
    y = scipy.sparse.random(100, 20, 0.2).tocsc()

    # Full:
    out = delayedarray.to_sparse_array(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == y.toarray()).all()

    # Consecutive subset.
    subs = (range(10, 80), range(5, 15))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y[numpy.ix_(*subs)].toarray()).all()

    # Non-consecutive subset.
    subs = (range(10, 80, 5), range(5, 15, 2))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y[numpy.ix_(*subs)].toarray()).all()

    # Empty.
    empty = scipy.sparse.random(100, 20, 0).tocsc()
    out = delayedarray.to_sparse_array(empty)
    assert (numpy.array(out) == numpy.zeros((100, 20))).all()

    # Nosub.
    out = delayedarray.extract_sparse_array(y, ([], [1]))
    assert out.shape == (0, 1)
    out = delayedarray.extract_sparse_array(y, ([0], []))
    assert out.shape == (1, 0)


def test_extract_sparse_array_csr_matrix():
    y = scipy.sparse.random(100, 20, 0.2).tocsr()

    # Full:
    out = delayedarray.to_sparse_array(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == y.toarray()).all()

    # Consecutive subset.
    subs = (range(10, 80), range(5, 15))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y[numpy.ix_(*subs)].toarray()).all()

    # Non-consecutive subset.
    subs = (range(10, 80, 5), range(5, 15, 2))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y[numpy.ix_(*subs)].toarray()).all()

    # Empty.
    empty = scipy.sparse.random(100, 20, 0).tocsr()
    out = delayedarray.to_sparse_array(empty)
    assert (numpy.array(out) == numpy.zeros((100, 20))).all()

    # Nosub.
    out = delayedarray.extract_sparse_array(y, ([], [1]))
    assert out.shape == (0, 1)
    out = delayedarray.extract_sparse_array(y, ([0], []))
    assert out.shape == (1, 0)


def test_extract_sparse_array_coo_matrix():
    y = scipy.sparse.random(100, 20, 0.2).tocoo()

    # Full:
    out = delayedarray.to_sparse_array(y)
    assert isinstance(out, delayedarray.SparseNdarray)
    assert (numpy.array(out) == y.toarray()).all()

    # Consecutive subset.
    subs = (range(10, 80), range(5, 15))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y.toarray()[numpy.ix_(*subs)]).all()

    # Non-consecutive subset.
    subs = (range(10, 80, 5), range(5, 15, 2))
    out = delayedarray.extract_sparse_array(y, subs)
    assert (numpy.array(out) == y.toarray()[numpy.ix_(*subs)]).all()

    # Empty.
    empty = scipy.sparse.random(100, 20, 0).tocoo()
    out = delayedarray.to_sparse_array(empty)
    assert (numpy.array(out) == numpy.zeros((100, 20))).all()

    # Nosub.
    out = delayedarray.extract_sparse_array(y, ([], [1]))
    assert out.shape == (0, 1)
    out = delayedarray.extract_sparse_array(y, ([0], []))
    assert out.shape == (1, 0)
