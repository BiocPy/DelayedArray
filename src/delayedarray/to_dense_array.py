from functools import singledispatch
from typing import Any
import numpy

from .extract_dense_array import extract_dense_array
from ._subset import _spawn_indices


@singledispatch
def to_dense_array(x: Any) -> numpy.ndarray:
    """
    Extract ``x`` as a dense NumPy array. The default method simply calls
    :py:func:`~delayedarray.extract_dense_array.extract_dense_array` with
    ``subset`` set to the full extent of all dimensions.

    Args:
        x: Any array-like object.

    Returns:
        NumPy array contains the full contents of ``x``. This may be masked.
    """
    return extract_dense_array(x, _spawn_indices(x.shape))
