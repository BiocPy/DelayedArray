from functools import singledispatch
from numpy import array, ndarray, ix_

from ._subset import _spawn_indices, _is_subset_noop
from .SparseNdarray import SparseNdarray, _extract_sparse_array_from_SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_sparse_array(x: Any, subset: Optional[Tuple[Sequence[int]]] = None) -> SparseNdarray:
    """Extract the contents of ``x`` (or a subset thereof) into a
    :py:class:`~delayedarray.SparseNdarray.SparseNdarray`. This should only be
    used for ``x`` where :py:meth:`~delayedarray.utils.is_sparse` returns True.

    Args:
        x: 
            Any object with a ``__DelayedArray_extract_sparse__`` method that
            accepts a non-None ``subset`` and returns a ``SparseNdarray``
            corresponding to the outer product of the subsets.

        subset: 
            Tuple of length equal to the number of dimensions, each containing
            a sorted and unique sequence of integers specifying the elements of
            each dimension to extract. If None, all elements are extracted from
            all dimensions.

    Returns:
        SparseNdarray for the requested subset. This may be a view so callers
        should create a copy if they intend to modify it.
    """

    if hasattr(x, "__DelayedArray_extract_sparse__"):
        if subset is None:
            subset = _spawn_indices(x.shape)
        return x.__DelayedArray_extract__(subset)
    raise NotImplementedError("'extract_sparse_array(" + str(type(x)) + ")' has not yet been implemented") 


@extract_sparse_array.register
def extract_sparse_array_SparseNdarray(x: ndarray, subset: Optional[Tuple[Sequence[int]]] = None):
    if _is_subset_noop(x.shape, subset):
        subset = None
    if subset is None:
        return x
    else:
        return _extract_sparse_array_from_SparseNdarray(x, subset)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.
has_sparse = False
try:
    import scipy.sparse
    has_sparse = True
except:
    pass


if has_sparse:
    pass
