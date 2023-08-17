from functools import singledispatch
from typing import Sequence
from SparseNdarray import SparseNdarray
import numpy

@singledispatch
def extract_dense_array(x: Any, idx: Tuple[Sequence, ...]) -> numpy.ndarray:
    raise NotImplementedError(
        f"extract_dense_array is not supported for objects of class: {type(x)}"
    )

@singledispatch
def extract_sparse_array(x: Any, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    raise NotImplementedError(
        f"extract_sparse_array is not supported for objects of class: {type(x)}"
    )

@extract_sparse_array.register
def extract_sparse_array(x: SparseNdarray, idx: Tuple[Sequence, ...]) -> SparseNdArray:
    new_contents = None
    if x.__contents is not None:
        if len(x.shape) > 1:
            new_contents = _recursive_extract_sparse_array(x.__contents, idx, 0)
        else:
            new_contents = _extract_sparse_vector_to_sparse(x.__contents[0], x.__contents[1], idx[0])

    idims = [len(y) for y in idx]
    return SparseNdarray(shape = (*idims), contents = new_contents)


