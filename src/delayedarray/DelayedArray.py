from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .UnaryIsometricOpWithArgs import UnaryIsometricOpWithArgs
from .UnaryIsometricOpSimple import UnaryIsometricOpSimple
from .utils import sanitize_indices, sanitize_single_index
import numpy

def wrap_isometric_with_args(x, other, op, right):
    # TO DO: handle binary operations for DelayedArray 'other'.
    return DelayedArray(UnaryIsometricOpWithArgs(
        x._seed, 
        value=other,
        op=op,
        along=len(x.shape) - 1, 
        right=right
    ))


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
        converted = numpy.array2string(bits_and_pieces, separator=", ", prefix=prefix, suffix=suffix, threshold=0)
        return preamble + "\n" + converted

    # Adding in all the magic methods with an 'other' argument.
    def __add__(self, other):
        return wrap_isometric_with_args(self, other, op="+", right=True)

    def __radd__(self, other):
        return wrap_isometric_with_args(self, other, op="+", right=False)

    def __sub__(self, other):
        return wrap_isometric_with_args(self, other, op="-", right=True)

    def __sub__(self, other):
        return wrap_isometric_with_args(self, other, op="-", right=False)

    def __mul__(self, other):
        return wrap_isometric_with_args(self, other, op="*", right=True)

    def __mul__(self, other):
        return wrap_isometric_with_args(self, other, op="*", right=False)

    def __truediv__(self, other):
        return wrap_isometric_with_args(self, other, op="/", right=True)

    def __rtruediv__(self, other):
        return wrap_isometric_with_args(self, other, op="/", right=False)

    def __mod__(self, other):
        return wrap_isometric_with_args(self, other, op="%", right=True)

    def __rmod__(self, other):
        return wrap_isometric_with_args(self, other, op="%", right=False)

    def __floordiv__(self, other):
        return wrap_isometric_with_args(self, other, op="//", right=True)

    def __rfloordiv__(self, other):
        return wrap_isometric_with_args(self, other, op="//", right=False)

    def __pow__(self, other):
        return wrap_isometric_with_args(self, other, op="**", right=True)

    def __rpow__(self, other):
        return wrap_isometric_with_args(self, other, op="**", right=False)

    # Adding all the true unary magic methods.
    def __neg__(self):
        return wrap_isometric_with_args(self, 0, op="-", right=False)

    def __abs__(self):
        return DelayedArray(UnaryIsometricSimple(self._seed, op="abs"))

    def __ceil__(self):
        return DelayedArray(UnaryIsometricSimple(self._seed, op="ceil"))

    def __floor__(self):
        return DelayedArray(UnaryIsometricSimple(self._seed, op="floor"))

    def __trunc__(self):
        return DelayedArray(UnaryIsometricSimple(self._seed, op="trunc"))


@extract_dense_array.register
def _extract_dense_array_DelayedArray(x: DelayedArray) -> numpy.ndarray:
    return extract_dense_array(x._seed)


@extract_sparse_array.register
def _extract_sparse_array_DelayedArray(x: DelayedArray) -> SparseNdarray:
    return extract_sparse_array(x._seed)


@is_sparse.register
def _is_sparse_DelayedArray(x: DelayedArray) -> bool:
    return is_sparse(x._seed)
