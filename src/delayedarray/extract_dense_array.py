from functools import singledispatch
import numpy
from typing import Any, Optional, Tuple, Sequence
from biocutils.package_utils import is_package_installed

from ._subset import _spawn_indices, _is_subset_noop
from .SparseNdarray import SparseNdarray, _extract_dense_array_from_SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_dense_array(x: Any, subset: Optional[Tuple[Sequence[int], ...]] = None) -> numpy.ndarray:
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

        If :py:func:`~delayedarray.is_masked.is_masked` is True for ``x``, a NumPy
        ``MaskedArray`` is returned instead.
    """
    raise NotImplementedError("'extract_dense_array(" + str(type(x)) + ")' has not yet been implemented") 


@extract_dense_array.register
def extract_dense_array_ndarray(x: numpy.ndarray, subset: Optional[Tuple[Sequence[int], ...]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    if _is_subset_noop(x.shape, subset):
        subset = None
    if subset is None:
        tmp = x
    else:
        tmp = x[numpy.ix_(*subset)]
    return _sanitize_to_fortran(tmp)


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
        # Don't use asfortranarray, as this strips any masks.
        return x.astype(x.dtype, order="F", casting="no", copy=False)


if is_package_installed("scipy"):
    import scipy.sparse as sp

    def _extract_dense_array_sparse(x, subset: Optional[Tuple[Sequence[int], ...]] = None):
        if _is_subset_noop(x.shape, subset):
            tmp = x
        else:
            # This just drops any masking on the scipy data; so, not our fault.
            # I am inclined to believe that scipy.sparse does not support
            # masked arrays, which is fine with me.
            tmp = x[numpy.ix_(*subset)]
        return tmp.toarray(order="F")

    @extract_dense_array.register
    def extract_dense_array_csc_matrix(x: sp.csc_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    @extract_dense_array.register
    def extract_dense_array_csr_matrix(x: sp.csr_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    @extract_dense_array.register
    def extract_dense_array_coo_matrix(x: sp.coo_matrix, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)

    def extract_dense_array_sparse_array(x, subset: Optional[Tuple[Sequence[int], ...]] = None):
        """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
        return _extract_dense_array_sparse(x, subset)
    
    try:
        extract_dense_array.register(sp.sparray, extract_dense_array_sparse_array)
    except Exception:
        pass

