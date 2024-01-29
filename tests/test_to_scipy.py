import delayedarray
import numpy
import scipy.sparse

from utils import simulate_SparseNdarray


def test_to_scipy_csc_matrix():
    test_shape = (100, 150)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_csc_matrix(y)
    assert isinstance(z, scipy.sparse.csc_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_csc_matrix(delayedarray.wrap(y))
    assert isinstance(z, scipy.sparse.csc_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()


def test_to_scipy_csr_matrix():
    test_shape = (150, 80)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_csr_matrix(y)
    assert isinstance(z, scipy.sparse.csr_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_csr_matrix(delayedarray.wrap(y))
    assert isinstance(z, scipy.sparse.csr_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()


def test_to_scipy_coo_matrix():
    test_shape = (70, 90)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_coo_matrix(y)
    assert isinstance(z, scipy.sparse.coo_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_coo_matrix(delayedarray.wrap(y))
    assert isinstance(z, scipy.sparse.coo_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()
