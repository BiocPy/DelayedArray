from typing import Sequence, Tuple, Union

from numpy import array2string, dtype, get_printoptions, ndarray

from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .Subset import Subset
from .UnaryIsometricOpSimple import UnaryIsometricOpSimple
from .UnaryIsometricOpWithArgs import UnaryIsometricOpWithArgs
from .utils import sanitize_indices

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def wrap_isometric_with_args(x, other, op, right):
    # TO DO: handle binary operations for DelayedArray 'other'.
    return DelayedArray(
        UnaryIsometricOpWithArgs(
            x._seed, value=other, op=op, along=len(x.shape) - 1, right=right
        )
    )


translate_ufunc_to_op_with_args = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "floor_divide": "//",
    "remainder": "%",
    "power": "**",
}

translate_ufunc_to_op_simple = set(
    [
        "log",
        "log1p",
        "log2",
        "log10",
        "exp",
        "expm1",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "ceil",
        "floor",
        "trunc",
        "sign",
    ]
)


class DelayedArray:
    """Array containing delayed operations.

    This allows users to efficiently operate on large matrices without actually
    evaluating the operation or creating new copies.

    Attributes:
        seed:
            Any array-like object with the ``shape`` and ``dtype`` properties.

            This should also have methods for the
            :py:meth:`~delayedarray.interface.is_sparse`,
            :py:meth:`~delayedarray.interface.extract_dense_array`,
            and (if ``is_sparse`` may return True)
            :py:meth:`~delayedarray.interface.extract_sparse_array` generics.
    """

    def __init__(self, seed):
        self._seed = seed

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the delayed array.

        Returns:
            Tuple[int, ...]: Tuple of integers containing the array shape along
            each dimension.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the elements in the array.

        Returns:
            dtype: Type of the NumPy array containing the values of the non-zero elements.
        """
        return self._seed.dtype

    def __repr__(self):
        total = 1
        for s in self._seed.shape:
            total *= s

        preamble = "<" + " x ".join([str(x) for x in self._seed.shape]) + ">"
        if is_sparse(self._seed):
            preamble += " sparse"
        preamble += " DelayedArray object of type '" + self._seed.dtype.name + "'"

        ndims = len(self._seed.shape)
        if total <= get_printoptions()["threshold"]:
            full_indices = [slice(None)] * ndims
            bits_and_pieces = extract_dense_array(self._seed, (*full_indices,))
            return preamble + "\n" + repr(bits_and_pieces)

        indices = []
        edge_size = get_printoptions()["edgeitems"]
        for d in range(ndims):
            extent = self._seed.shape[d]
            if extent > edge_size * 2:
                indices.append(
                    list(range(edge_size + 1)) + list(range(extent - edge_size, extent))
                )
            else:
                indices.append(slice(None))

        bits_and_pieces = extract_dense_array(self._seed, (*indices,))
        converted = array2string(bits_and_pieces, separator=", ", threshold=0)
        return preamble + "\n" + converted

    # For NumPy:
    def __array__(self):
        full_indices = sanitize_indices([slice(None)] * len(self.shape), self.shape)
        return extract_dense_array(self._seed, (*full_indices,))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Interface with NumPy array methods.

        Usage:

        .. code-block:: python

            np.sqrt(delayed_array)

        Returns:
            An object with the same type as caller.
        """

        if ufunc.__name__ in translate_ufunc_to_op_with_args:
            # This is required to support situations where the NumPy array is on
            # the LHS, such that the ndarray method gets called first.
            op = translate_ufunc_to_op_with_args[ufunc.__name__]
            first_is_da = isinstance(inputs[0], DelayedArray)
            da = inputs[1 - int(first_is_da)]
            v = inputs[int(first_is_da)]
            return wrap_isometric_with_args(da, v, op=op, right=first_is_da)
        elif ufunc.__name__ in translate_ufunc_to_op_simple:
            return DelayedArray(UnaryIsometricOpSimple(inputs[0], op=ufunc.__name__))
        elif ufunc.__name__ == "absolute":
            return DelayedArray(UnaryIsometricOpSimple(inputs[0], op="abs"))

        raise NotImplementedError(f"'{ufunc.__name__}' is not implemented!")

    def __add__(self, other):
        return wrap_isometric_with_args(self, other, op="+", right=True)

    def __radd__(self, other):
        return wrap_isometric_with_args(self, other, op="+", right=False)

    def __sub__(self, other):
        return wrap_isometric_with_args(self, other, op="-", right=True)

    def __rsub__(self, other):
        return wrap_isometric_with_args(self, other, op="-", right=False)

    def __mul__(self, other):
        return wrap_isometric_with_args(self, other, op="*", right=True)

    def __rmul__(self, other):
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

    def __neg__(self):
        return wrap_isometric_with_args(self, 0, op="-", right=False)

    def __abs__(self):
        return DelayedArray(UnaryIsometricOpSimple(self._seed, op="abs"))

    def __getitem__(self, args: Tuple[Union[slice, Sequence], ...]) -> "DelayedArray":
        """Create a delayed subset wrapper around this array.

        Args:
            args (Tuple[Union[slice, Sequence], ...]): A :py:class`tuple` defining the
                positions of the array to access along each dimension.

                Each element in ``args`` may be a :py:func:`slice` object or
                a list of integer indices. The length of the tuple
                must not exceed the number of dimensions in the array.

        Raises:
            ValueError: If ``args`` contain more dimensions than the shape of the array.

        Returns:
            A DelayedArray containing a delayed subset operation.
        """
        return DelayedArray(Subset(self._seed, args))


@extract_dense_array.register
def _extract_dense_array_DelayedArray(
    x: DelayedArray, idx: Tuple[Sequence, ...]
) -> ndarray:
    return extract_dense_array(x._seed, idx)


@extract_sparse_array.register
def _extract_sparse_array_DelayedArray(
    x: DelayedArray, idx: Tuple[Sequence, ...]
) -> SparseNdarray:
    return extract_sparse_array(x._seed, idx)


@is_sparse.register
def _is_sparse_DelayedArray(x: DelayedArray) -> bool:
    return is_sparse(x._seed)
