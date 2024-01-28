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
        Whether ``x`` contains masked values. If no method is defined
        for ``x``, False is returned by default.
    """
    return False


@is_masked.register
def is_masked_MaskedArray(x: numpy.ma.core.MaskedArray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return True


@is_masked.register
def is_masked_SparseNdarray(x: SparseNdarray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return x._is_masked
