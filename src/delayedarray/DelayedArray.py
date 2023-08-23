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
    """Array containing delayed operations. This is equivalent to the class of the same name from 
    the `R/Bioconductor package <https://bioconductor.org/packages/DelayedArray>`_ of the same name.
    It allows users to efficiently operate on large matrices without actually evaluating the
    operation or creating new copies; instead, the operations will transparently return another DelayedArray instance
    containing the delayed operations, which can be realized by calling :py:meth:`~numpy.array` or related methods.

    Attributes:
        seed:
            Any array-like object that satisfies the "seed contract".
            This means that it has the :py:attr:`~shape` and :py:attr:`~dtype` properties.
            It should also have methods for the
            :py:meth:`~delayedarray.interface.is_sparse`,
            :py:meth:`~delayedarray.interface.extract_dense_array`,
            and (if :py:meth:`~delayedarray.interface.is_sparse` could return True)
            :py:meth:`~delayedarray.interface.extract_sparse_array` generics.
    """

    def __init__(self, seed):
        self._seed = seed

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the DelayedArray.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the DelayedArray.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the elements in the DelayedArray.

        Returns:
            dtype: NumPy type of the values.
        """
        return self._seed.dtype

    def __repr__(self) -> str:
        """Pretty-print this DelayedArray. This uses :py:meth:`~numpy.array2string` and responds to all of its options.

        Returns:
            str: String containing a prettified display of the array contents.
        """
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
    def __array__(self) -> ndarray:
        """Convert a DelayedArray to a NumPy array.

        Returns:
            ndarray: Array of the same type as :py:attr:`~dtype` and shape as :py:attr:`~shape`.
            This is guaranteed to be in C-contiguous order and to not be a view on other data.
        """
        full_indices = sanitize_indices([slice(None)] * len(self.shape), self.shape)
        return extract_dense_array(self._seed, (*full_indices,))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "DelayedArray":
        """Interface with NumPy array methods.
        This is used to implement mathematical operations like NumPy's :py:meth:`~numpy.log`,
        or to override operations between NumPy class instances and DelayedArrays where the former is on the left hand side.
        Check out the NumPy's ``__array_ufunc__`` `documentation <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__>`_ for more details.

        Returns:
            DelayedArray: A DelayedArray instance containing the requested delayed operation.
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

    def __add__(self, other) -> "DelayedArray":
        """Add something to the right-hand-side of a DelayedArray.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed addition operation.
        """
        return wrap_isometric_with_args(self, other, op="+", right=True)

    def __radd__(self, other) -> "DelayedArray":
        """Add something to the left-hand-side of a DelayedArray.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed addition operation.
        """
        return wrap_isometric_with_args(self, other, op="+", right=False)

    def __sub__(self, other) -> "DelayedArray":
        """Subtract something from the right-hand-side of a DelayedArray.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed subtraction operation.
        """
        return wrap_isometric_with_args(self, other, op="-", right=True)

    def __rsub__(self, other):
        """Subtract a DelayedArray from something else.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed subtraction operation.
        """
        return wrap_isometric_with_args(self, other, op="-", right=False)

    def __mul__(self, other):
        """Multiply a DelayedArray with something on the right hand side.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed multiplication operation.
        """
        return wrap_isometric_with_args(self, other, op="*", right=True)

    def __rmul__(self, other):
        """Multiply a DelayedArray with something on the left hand side.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed multiplication operation.
        """
        return wrap_isometric_with_args(self, other, op="*", right=False)

    def __truediv__(self, other):
        """Divide a DelayedArray by something.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed division operation.
        """
        return wrap_isometric_with_args(self, other, op="/", right=True)

    def __rtruediv__(self, other):
        """Divide something by a DelayedArray.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed division operation.
        """
        return wrap_isometric_with_args(self, other, op="/", right=False)

    def __mod__(self, other):
        """Take the remainder after dividing a DelayedArray by something.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed modulo operation.
        """
        return wrap_isometric_with_args(self, other, op="%", right=True)

    def __rmod__(self, other):
        """Take the remainder after dividing something by a DelayedArray.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed modulo operation.
        """
        return wrap_isometric_with_args(self, other, op="%", right=False)

    def __floordiv__(self, other):
        """Divide a DelayedArray by something and take the floor.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed floor division operation.
        """
        return wrap_isometric_with_args(self, other, op="//", right=True)

    def __rfloordiv__(self, other):
        """Divide something by a DelayedArray and take the floor.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed floor division operation.
        """
        return wrap_isometric_with_args(self, other, op="//", right=False)

    def __pow__(self, other):
        """Raise a DelayedArray to the power of something.

        Args:
            other:
                A numeric scalar or a NumPy array of length equal to the extent of the last dimension of the DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed power operation.
        """
        return wrap_isometric_with_args(self, other, op="**", right=True)

    def __rpow__(self, other):
        """Raise something to the power of the contents of a DelayedArray.

        Args:
            other:
                A numeric scalar.
                In theory, this could also be a NumPy array of length equal to the extent of the last dimension of the DelayedArray,
                but that is usually handled via :py:meth:`~__array_ufunc__`.

        Returns:
            DelayedArray: A DelayedArray containing the delayed power operation.
        """
        return wrap_isometric_with_args(self, other, op="**", right=False)

    def __neg__(self):
        """Negate the contents of a DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed negation.
        """
        return wrap_isometric_with_args(self, 0, op="-", right=False)

    def __abs__(self):
        """Take the absolute value of the contents of a DelayedArray.

        Returns:
            DelayedArray: A DelayedArray containing the delayed absolute value operation.
        """
        return DelayedArray(UnaryIsometricOpSimple(self._seed, op="abs"))

    def __getitem__(self, args: Tuple[Union[slice, Sequence], ...]) -> "DelayedArray":
        """Take a subset of this DelayedArray. Unlike NumPy, the subset will be an outer product of the per-dimension
        indices defined in ``args``; this aligns with the behavior of subsetting in R, and is equivalent to using
        NumPy's :py:meth:`~numpy.ix_` function.

        Args:
            args (Tuple[Union[slice, Sequence], ...]):
                A :py:class:`tuple` of length equal to the dimensionality of this DelayedArray.
                Each entry should contain a sequence of integer indices (e.g., a list, :py:class:`~numpy.ndarray` or :py:func:`slice`),
                specifying the elements of the corresponding dimension to extract.

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
