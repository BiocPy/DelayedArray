from functools import singledispatch
from numpy import array, ndarray, ix_, asfortranarray
from typing import Any, Optional, Tuple, Sequence

from ._subset import _spawn_indices, _is_subset_noop
from .SparseNdarray import SparseNdarray, _extract_dense_array_from_SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_dense_array(x: Any, subset: Optional[Tuple[Sequence[int], ...]] = None) -> ndarray:
    """Extract the realized contents (or a subset thereof) into a dense NumPy
    array with Fortran storage order, i.e., earlier dimensions change fastest. 

    Args:
        x: 
            Any array-like object.

        subset: 
            Tuple of length equal to the number of dimensions, each containing
            a sorted and unique sequence of integers specifying the elements of
            each dimension to extract. If None, all elements are extracted from
            all dimensions.

    Returns:
        NumPy array with Fortran storage order. This may be a view so callers should
        create a copy if they intend to modify it.
    """
    raise NotImplementedError("'extract_dense_array(" + str(type(x)) + ")' has not yet been implemented") 


@extract_dense_array.register
def extract_dense_array_ndarray(x: ndarray, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    if _is_subset_noop(x.shape, subset):
        subset = None
    if subset is None:
        tmp = x
    else:
        tmp = x[ix_(*subset)]
    return array(tmp, dtype=tmp.dtype, order="F", copy=False)


@extract_dense_array.register
def extract_dense_array_SparseNdarray(x: SparseNdarray, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    if subset is None:
        subset = _spawn_indices(x.shape)
    return _extract_dense_array_from_SparseNdarray(x, subset)


def _sanitize_to_fortran(x):
    if x.flags.f_contiguous:
        return x
    else: 
        return asfortranarray(x)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.
has_sparse = False
try:
    import scipy.sparse
    has_sparse = True
except:
    pass


if has_sparse:
    def _extract_dense_array_sparse(x, subset: Optional[Tuple[Sequence[int], ...]] = None):
        if _is_subset_noop(x.shape, subset):
            tmp = x
        else:
            tmp = x[ix_(*subset)]
        return tmp.toarray(order="F")

    @extract_dense_array.register
    def extract_dense_array_csc_matrix(x: scipy.sparse.csc_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    @extract_dense_array.register
    def extract_dense_array_csr_matrix(x: scipy.sparse.csr_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    @extract_dense_array.register
    def extract_dense_array_coo_matrix(x: scipy.sparse.coo_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    @extract_dense_array.register
    def extract_dense_array_sparse_array(x: scipy.sparse.sparray, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)
