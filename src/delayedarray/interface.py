import copy
from functools import singledispatch
from typing import Sequence, Tuple

from numpy import ndarray

from .SparseNdarray import (
    SparseNdarray,
    _extract_dense_array_from_SparseNdarray,
    _extract_sparse_array_from_SparseNdarray,
)

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def extract_dense_array(x, idx: Tuple[Sequence, ...]) -> ndarray:
    """Extract slice from a dense array.

    Args:
        x: Array to slice.

            ``x`` may be either a :py:class.`~numpy.ndarray` or
            :py:class.`~delayedarray.SparseNdarray.SparseNdarray`.

        idx (Tuple[Sequence, ...]): Indices to slice, must be less
            than or equal to the number of dimensions in the dense matrix.

    Raises:
        NotImplementedError: When ``x`` is not an supported type.

    Returns:
        ndarray: A sliced :py:class.`numpy.ndarray`.
    """
    raise NotImplementedError(
        f"extract_dense_array is not supported for '{type(x)}' objects"
    )


@extract_dense_array.register
def extract_dense_array_ndarray(x: ndarray, idx: Tuple[Sequence, ...]) -> ndarray:
    return copy.deepcopy(x[(..., *idx)])


@extract_dense_array.register
def _extract_dense_array_SparseNdarray(
    x: SparseNdarray, idx: Tuple[Sequence, ...]
) -> ndarray:
    return _extract_dense_array_from_SparseNdarray(x, idx)


@singledispatch
def extract_sparse_array(x, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    """Extract slice from a sparse array.

    Args:
        x: Array to slice.

            ``x`` may be either a :py:class.`~scipy.sparse.spmatrix` or
            :py:class.`~delayedarray.SparseNdarray.SparseNdarray`.

        idx (Tuple[Sequence, ...]): Indices to slice, must be less
            than or equal to the number of dimensions in the sparse array.

    Raises:
        NotImplementedError: When ``x`` is not an supported type.
    Returns:
        SparseNdarray: A sliced sparse array object.
    """
    raise NotImplementedError(
        f"extract_sparse_array is not supported for '{type(x)}' objects"
    )


@extract_sparse_array.register
def _extract_sparse_array_SparseNdarray(
    x: SparseNdarray, idx: Tuple[Sequence, ...]
) -> SparseNdarray:
    return _extract_sparse_array_from_SparseNdarray(x, idx)


@singledispatch
def is_sparse(x) -> bool:
    """Check if ``x`` represents a sparse array.

    Args:
        x: Array to check.

    Returns:
        bool: True if ``x`` is a sparse array.
    """
    return False


@is_sparse.register
def _is_sparse_ndarray(x: ndarray) -> bool:
    return False


@is_sparse.register
def _is_sparse_SparseNdarray(x: SparseNdarray) -> bool:
    return True
