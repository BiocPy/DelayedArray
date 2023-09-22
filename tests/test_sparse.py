import delayedarray
import numpy
import scipy.sparse


def test_sparse():
    y = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.DelayedArray(y)

    out = delayedarray.extract_array(x)
    assert isinstance(out, numpy.ndarray) is False
    assert delayedarray.chunk_shape(out) == (1, 3)


def test_sparse_arithmetic():
    y = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.DelayedArray(y)

    z = x + 1
    out = delayedarray.extract_array(z)
    assert isinstance(out, numpy.ndarray) is True

    v = numpy.random.rand(3)
    z = x * v
    out = delayedarray.extract_array(z)
    assert (y.toarray() * v == numpy.array(out)).all()

    z = x / v
    out = delayedarray.extract_array(z)
    assert numpy.allclose(y.toarray() / v, out.toarray())


def test_sparse_math():
    y = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.DelayedArray(y)
    z = numpy.log1p(x)
    out = delayedarray.extract_array(z)
    assert isinstance(out, numpy.ndarray) is False
    assert (numpy.log1p(y.toarray()) == out.toarray()).all()


def test_sparse_subset():
    y = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.DelayedArray(y)

    z = x[1:3, [0, 2]]
    out = delayedarray.extract_array(z)
    assert isinstance(out, numpy.ndarray) is False
    assert (y.toarray()[1:3, [0, 2]] == out.toarray()).all()

    assert (y.toarray()[1, :] == x[1, :]).all()
    assert (y.toarray()[:, 0] == x[:, 0]).all()


def test_sparse_transpose():
    y = scipy.sparse.csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    x = delayedarray.DelayedArray(y)
    z = numpy.transpose(x)
    out = delayedarray.extract_array(z)
    assert isinstance(out, numpy.ndarray) is False
    assert (numpy.transpose(y.toarray()) == out.toarray()).all()
