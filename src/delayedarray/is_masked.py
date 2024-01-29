from functools import singledispatch
from typing import Any
import numpy
from biocutils.package_utils import is_package_installed

from .SparseNdarray import SparseNdarray


@singledispatch
def is_masked(x: Any) -> bool:
    """
    Determine whether an array-like object contains masked values,
    equivalent to those in NumPy's ``MaskedArray`` class.

    Args:
        x: Any array-like object.

    Returns:
        Whether ``x`` contains masked values.
    """
    raise NotImplementedError("'is_masked(" + str(type(x)) + ")' has not yet been implemented") 


@is_masked.register
def is_masked_ndarray(x: numpy.ndarray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return False


@is_masked.register
def is_masked_MaskedArray(x: numpy.ma.core.MaskedArray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return True


@is_masked.register
def is_masked_SparseNdarray(x: SparseNdarray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return x._is_masked

# If scipy is installed, we add all the methods for the various scipy.sparse
# matrices. Currently, it seems like scipy's sparse matrices are not intended
# to be masked, seeing as how any subsetting discards the masks, e.g.,
#
# >>> y = (scipy.sparse.random(1000, 200, 0.1)).tocsr()
# >>> y.data = numpy.ma.MaskedArray(y.data, y.data > 0.5)
# >>> y[0:5,:].data # gives back a regulary NumPy array.
#
# So we won't bother capturing the mask state (as of scipy 1.11.1).

if is_package_installed("scipy"):
    import scipy.sparse as sp

    @is_masked.register
    def is_masked_csc_matrix(x: sp.csc_matrix):
        """See :py:meth:`~delayedarray.is_masked.is_masked`."""
        return False

    @is_masked.register
    def is_masked_csr_matrix(x: sp.csr_matrix):
        """See :py:meth:`~delayedarray.is_masked.is_masked`."""
        return False

    @is_masked.register
    def is_masked_coo_matrix(x: sp.coo_matrix):
        """See :py:meth:`~delayedarray.is_masked.is_masked`."""
        return False
