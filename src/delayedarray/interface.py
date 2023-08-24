import copy
from functools import singledispatch
from typing import Sequence, Tuple

from numpy import ix_, ndarray

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
    """Extract a dense array from a seed. This uses the outer product of the indices specified in `idx`.

    Args:
        x: Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        idx (Tuple[Sequence, ...]):
            Tuple of length equal to the number of dimensions in ``x``.
            Each entry should be a sequence of sorted and unique non-negative integers that are less than the extent of the corresponding dimension of ``x``,
            specifying the indices of the dimension to extract.

    Raises:
        NotImplementedError: When ``x`` is not an supported type.

    Returns:
        ndarray: An :py:class:`~numpy.ndarray` containing the dense contents of ``x`` at the requested indices.
        This is guaranteed to be in C-contiguous layout and to not be a view.
    """
    raise NotImplementedError(
        f"extract_dense_array is not supported for '{type(x)}' objects"
    )


@extract_dense_array.register
def _extract_dense_array_ndarray(x: ndarray, idx: Tuple[Sequence, ...]) -> ndarray:
    return copy.deepcopy(x[ix_(*idx)])


@extract_dense_array.register
def _extract_dense_array_SparseNdarray(
    x: SparseNdarray, idx: Tuple[Sequence, ...]
) -> ndarray:
    return _extract_dense_array_from_SparseNdarray(x, idx)


@singledispatch
def extract_sparse_array(x, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    """Extract a sparse array from a seed. This uses the outer product of the indices specified in `idx`.

    Args:
        x: Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        idx (Tuple[Sequence, ...]):
            Tuple of length equal to the number of dimensions in ``x``.
            Each entry should be a sequence of sorted and unique non-negative integers that are less than the extent of the corresponding dimension of ``x``,
            specifying the indices of the dimension to extract.

    Raises:
        NotImplementedError: When ``x`` is not an supported type.

    Returns:
        SparseNdarray: A ``SparseNdarray`` containing the sparse contents of ``x`` at the requested indices.
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
        x: Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

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
