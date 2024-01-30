from functools import singledispatch
from typing import Any, Tuple
from numpy import ndarray
from biocutils.package_utils import is_package_installed

from .SparseNdarray import SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def chunk_shape(x: Any) -> Tuple[int, ...]:
    """
    Get the dimensions of the array chunks. These define the preferred
    intervals with which to iterate over the array in each dimension, usually
    reflecting a particular layout on disk or in memory. The extent of each
    chunk dimension should be positive and less than that of the array's;
    except for zero-length dimensions, in which case the chunk's extent should
    be greater than the array (typically 1 to avoid divide by zero errors). 

    Args:
        x: An array-like object.
    
    Returns:
        Tuple of integers containing the shape of the chunk. If no method
        is defined for ``x``, an all-1 tuple is returned under the assumption
        that any element of any dimension can be accessed efficiently.
    """
    raw = [1] * len(x.shape)
    return (*raw,)


@chunk_shape.register
def chunk_shape_ndarray(x: ndarray):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    sh = list(x.shape)
    if x.flags.f_contiguous:
        for i in range(1, len(sh)):
            sh[i] = 1
    else:
        # Not sure how to deal with strided views here; not even sure how
        # to figure that out from NumPy flags. Guess we should just assume
        # that it's C-contiguous, given that most things are.
        for i in range(len(sh) - 1):
            sh[i] = 1
    return (*sh,)


@chunk_shape.register
def chunk_shape_SparseNdarray(x: SparseNdarray):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    chunks = [1] * len(x.shape)
    chunks[0] = x.shape[0]
    return (*chunks,)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.

if is_package_installed("scipy"):
    import scipy.sparse as sp

    @chunk_shape.register
    def chunk_shape_csc_matrix(x: sp.csc_matrix):
        """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
        return (x.shape[0], 1)


    @chunk_shape.register
    def chunk_shape_csr_matrix(x: sp.csr_matrix):
        """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
        return (1, x.shape[1])


    @chunk_shape.register
    def chunk_shape_coo_matrix(x: sp.coo_matrix):
        """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
        return x.shape # ???? well, let's just do our best.
