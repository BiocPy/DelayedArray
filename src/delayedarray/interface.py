from functools import singledispatch
from typing import Sequence, Tuple, Any
from .SparseNdarray import (
    SparseNdarray, 
    _extract_sparse_array_from_SparseNdarray,
    _extract_dense_array_from_SparseNdarray
)
import numpy
import copy

@singledispatch
def extract_dense_array(x: Any, idx: Tuple[Sequence, ...]) -> numpy.ndarray:
    raise NotImplementedError(
        f"extract_dense_array is not supported for '{type(x)}' objects"
    )

@singledispatch
def extract_sparse_array(x: Any, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    raise NotImplementedError(
        f"extract_sparse_array is not supported for '{type(x)}' objects"
    )

@extract_dense_array.register
def extract_dense_array_ndarray(x: numpy.ndarray, idx: Tuple[Sequence, ...]) -> numpy.ndarray:
    return copy.deepcopy(x[(..., *idx)])

@extract_dense_array.register
def extract_dense_array_SparseNdarray(x: SparseNdarray, idx: Tuple[Sequence, ...]) -> numpy.ndarray:
    return _extract_dense_array_from_SparseNdarray(x, idx)

@extract_sparse_array.register
def extract_sparse_array_SparseNdarray(x: SparseNdarray, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    return _extract_sparse_array_from_SparseNdarray(x, idx)

@singledispatch
def is_sparse(x: Any) -> bool:
    return False

@is_sparse.register
def is_sparse_ndarray(x: numpy.ndarray) -> bool:
    return False

@is_sparse.register
def is_sparse_SparseNdarray(x: SparseNdarray) -> bool:
    return True
