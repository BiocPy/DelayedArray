from functools import singledispatch
from typing import Any
from biocutils.package_utils import is_package_installed

from .SparseNdarray import SparseNdarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def is_sparse(x: Any) -> bool:
    """Determine whether an array-like object contains sparse data.

    Args:
        x: Any array-like object.

    Returns:
        Whether ``x`` contains sparse data. If no method is defined
        for ``x``, False is returned by default.
    """
    return False


@is_sparse.register
def is_sparse_SparseNdarray(x: SparseNdarray):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return True


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.

if is_package_installed("scipy"):
    import scipy.sparse as sp

    @is_sparse.register
    def is_sparse_csc_matrix(x: sp.csc_matrix):
        """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
        return True

    @is_sparse.register
    def is_sparse_csr_matrix(x: sp.csr_matrix):
        """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
        return True

    @is_sparse.register
    def is_sparse_coo_matrix(x: sp.coo_matrix):
        """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
        return True
