from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .utils import sanitize_indices, sanitize_single_index
import numpy
import textwrap

class DelayedArray:
    def __init__(self, seed):
        self._seed = seed

    @property
    def shape(self):
        return self._seed.shape

    @property
    def dtype(self):
        return self._seed.dtype 

    def __repr__(self):
        total = 1
        for s in self._seed.shape:
            total *= s

        preamble = "<" + ' x '.join([str(x) for x in self._seed.shape]) + ">"
        if is_sparse(self._seed):
            preamble += " sparse"
        preamble += " DelayedArray object of type '" + self._seed.dtype.name + "'"

        ndims = len(self._seed.shape)
        if total <= numpy.get_printoptions()['threshold']:
            full_indices = [slice(None)] * ndims
            bits_and_pieces = extract_dense_array(self._seed, (*full_indices,))
            return preamble + "\n" + repr(bits_and_pieces)

        indices = []
        edge_size = numpy.get_printoptions()["edgeitems"]
        for d in range(ndims):
            extent = self._seed.shape[d]
            if extent > edge_size * 2:
                indices.append(list(range(edge_size + 1)) + list(range(extent - edge_size, extent)))
            else:
                indices.append(slice(None))

        bits_and_pieces = extract_dense_array(self._seed, (*indices,))
        prefix = "array("
        if self._seed.dtype == numpy.float64:
            suffix = ")"
        else:
            suffix = ", dtype=numpy." + self._seed.dtype.name + ")"

        converted = numpy.array2string(bits_and_pieces, separator=", ", prefix=prefix, suffix=suffix, threshold=0)
        return preamble + "\n" + prefix + converted + suffix

@extract_dense_array.register
def _extract_dense_array_DelayedArray(x: DelayedArray) -> numpy.ndarray:
    return extract_dense_array(x._seed)

@extract_sparse_array.register
def _extract_sparse_array_DelayedArray(x: DelayedArray) -> SparseNdarray:
    return extract_sparse_array(x._seed)

@is_sparse.register
def _is_sparse_DelayedArray(x: DelayedArray) -> bool:
    return is_sparse(x._seed)



