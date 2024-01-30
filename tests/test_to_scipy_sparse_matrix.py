import delayedarray
import numpy
import scipy.sparse

from utils import simulate_SparseNdarray


def test_to_scipy_sparse_matrix_csc():
    test_shape = (100, 150)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_sparse_matrix(y, "csc")
    assert isinstance(z, scipy.sparse.csc_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_sparse_matrix(delayedarray.wrap(y), "csc")
    assert isinstance(z, scipy.sparse.csc_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()


def test_to_scipy_sparse_matrix_csr():
    test_shape = (150, 80)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_sparse_matrix(y, "csr")
    assert isinstance(z, scipy.sparse.csr_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_sparse_matrix(delayedarray.wrap(y), "csr")
    assert isinstance(z, scipy.sparse.csr_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()


def test_to_scipy_sparse_matrix_coo():
    test_shape = (70, 90)
    y = simulate_SparseNdarray(test_shape)
    z = delayedarray.to_scipy_sparse_matrix(y, "coo")
    assert isinstance(z, scipy.sparse.coo_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()

    z = delayedarray.to_scipy_sparse_matrix(delayedarray.wrap(y), "coo")
    assert isinstance(z, scipy.sparse.coo_matrix)
    assert (z.toarray() == delayedarray.to_dense_array(y)).all()
