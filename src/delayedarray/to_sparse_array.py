from functools import singledispatch
from typing import Any
import numpy

from ._subset import _spawn_indices
from .extract_sparse_array import extract_sparse_array
from .SparseNdarray import SparseNdarray


@singledispatch
def to_sparse_array(x: Any) -> SparseNdarray:
    """
    Convert ``x`` to a :py:class:`~delayedarray.SparseNdarray.SparseNdarray`.
    This calls :py:func:`~delayedarray.delayedarray.extract_sparse_array` with
    ``subset`` set to the full extent of all dimensions.

    Args:
        x: Any array-like object containing sparse data.

    Returns:
        ``SparseNdarray`` with the full contents of ``x``.
    """
    return extract_sparse_array(x, _spawn_indices(x.shape))
