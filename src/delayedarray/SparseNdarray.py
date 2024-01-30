from bisect import bisect_left
from typing import Callable, List, Optional, Sequence, Tuple, Union
from collections import namedtuple
import numpy
import copy

from ._isometric import translate_ufunc_to_op_simple, translate_ufunc_to_op_with_args, ISOMETRIC_OP_WITH_ARGS, _choose_operator, _infer_along_with_args
from ._subset import _spawn_indices, _getitem_subset_preserves_dimensions, _getitem_subset_discards_dimensions, _repr_subset
from ._mask import (
    _allocate_unmasked_ndarray, 
    _allocate_maybe_masked_ndarray, 
    _convert_to_unmasked_1darray, 
    _convert_to_maybe_masked_1darray,
    _concatenate_unmasked_ndarrays,
    _concatenate_maybe_masked_ndarrays
)
from ._statistics import array_mean, array_var, array_sum, _create_offset_multipliers

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class SparseNdarray:
    """The ``SparseNdarray``, as its name suggests, is a sparse n-dimensional
    array. It is inspired by the ``SVTArray`` class from the `DelayedArray
    R/Bioconductor package <https://bioconductor.org/packages/DelayedArray>`_.

    Internally, the ``SparseNdarray`` is represented as a nested list where
    each nesting level corresponds to a dimension in reverse order, i.e., the
    outer-most list corresponds to elements of the last dimension in ``shape``.
    The list at each level has length equal to the extent of its dimension,
    where each entry contains another list representing the contents of the
    corresponding element of that dimension. This recursion continues until the
    second dimension (i.e., the penultimate nesting level), where each entry
    instead contains ``(index, value)`` tuples. In effect, this is a tree where
    the non-leaf nodes are lists and the leaf nodes are tuples.

    Each ``(index, value)`` tuple represents a sparse vector for the
    corresponding element of the first dimension of the ``SparseNdarray``.
    ``index`` should be a :py:class:`~numpy.ndarray` of integers where entries
    are strictly increasing and less than the extent of the first dimension.
    All ``index`` objects in the same ``SparseNdarray`` should have the same
    ``dtype`` (defined by the ``index_dtype`` property). ``value`` may be any
    numeric/boolean :py:class:`~numpy.ndarray` but the ``dtype`` should be
    consistent across all ``value`` objects in the ``SparseNdarray``. If the
    array contains masked values, all ``value`` objects should be a
    ``MaskedArray``, otherwise they should be regular NumPy arrays.

    Any entry of any (nested) list may also be None, indicating that the
    corresponding element of the dimension contains no non-zero values. In
    fact, the entire tree may be None, indicating that there are no non-zero
    values in the entire array.

    For 1-dimensional arrays, the contents should be a single ``(index,
    value)`` tuple containing the sparse contents. This may also be None if
    there are no non-zero values in the array.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        contents,
        dtype: Optional[numpy.dtype] = None,
        index_dtype: Optional[numpy.dtype] = None,
        is_masked: Optional[bool] = None,
        check: bool = True,
    ):
        """
        Args:
            shape: 
                Tuple specifying the dimensions of the array.

            contents:
                For ``n``-dimensional arrays where ``n`` > 1, a nested list representing a
                tree where each leaf node is a tuple containing a sparse vector (or None).

                For 1-dimensional arrays, a tuple containing a sparse vector.

                Alternatively None, if the array is empty.

            dtype:
                NumPy type of the array values.
                If None, this is inferred from ``contents``.

            index_dtype:
                NumPy type of the array indices.
                If None, this is inferred from ``contents``.

            is_masked:
                Whether ``contents`` contains masked values.
                If None, this is inferred from ``contents``.

            check:
                Whether to check the consistency of the ``contents`` during construction.
                This can be set to False for speed.
        """

        self._shape = shape
        self._contents = contents
        ndim = len(shape)

        if dtype is None or index_dtype is None or is_masked is None:
            if contents is not None:
                if ndim > 1:
                    info = _peek(contents, dim=ndim-1)
                    if info is not None:
                        index_dtype0 = info[0]
                        dtype0 = info[1]
                        is_masked0 = info[2]
                    else:
                        dtype0 = None
                        index_dtype0 = None
                        is_masked0 = False
                else:
                    index_dtype0 = contents[0].dtype
                    dtype0 = contents[1].dtype
                    is_masked0 = numpy.ma.isMaskedArray(contents[1])

                if dtype is None:
                    dtype = dtype0
                if index_dtype is None:
                    index_dtype = index_dtype0
                if is_masked is None:
                    is_masked = is_masked0

            if dtype is None:
                raise ValueError("cannot infer 'dtype' from 'contents'")
            if index_dtype is None:
                raise ValueError("cannot infer 'index_dtype' from 'contents'")
            if is_masked is None:
                is_masked = False 

        self._dtype = dtype
        self._index_dtype = index_dtype
        self._is_masked = is_masked

        if check is True and contents is not None:
            payload = _CheckPayload(dtype=self._dtype, index_dtype=self._index_dtype, is_masked=self._is_masked, max_index=self._shape[0])
            if len(shape) > 1:
                _recursive_check(self._contents, self._shape, payload=payload, dim=ndim-1)
            else:
                _check_sparse_tuple(self._contents[0], self._contents[1], payload=payload)

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension.
        """
        return self._shape

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type of the values. 
        """
        return self._dtype


    @property
    def index_dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type of the indices.
        """
        return self._index_dtype


    @property
    def is_masked(self) -> bool:
        """
        Returns:
            Whether the values are masked.
        """
        return self._is_masked


    @property
    def contents(self):
        """Contents of the array. This is intended to be read-only and 
        should only be modified if you really know what you're doing.

        Returns:
            A nested list, for a n-dimensional array where n > 1.

            A tuple containing a sparse vector (i.e., indices and values), for a 1-dimensional array.

            Alternatively None, if the array contains no non-zero elements.
        """
        return self._contents


    def __repr__(self) -> str:
        """Pretty-print this ``SparseNdarray``. This uses
        :py:meth:`~numpy.array2string` and responds to all of its options.

        Returns:
            String containing a prettified display of the array contents.
        """
        preamble = "<" + " x ".join([str(x) for x in self._shape]) + ">"
        preamble += " " + type(self).__name__ + " object of type '" + self._dtype.name + "'"
        indices = _repr_subset(self._shape)
        bits_and_pieces = _extract_dense_array_from_SparseNdarray(self, indices)
        converted = numpy.array2string(bits_and_pieces, separator=", ", threshold=0)
        return preamble + "\n" + converted


    # For NumPy:
    def __array__(self) -> numpy.ndarray:
        """Convert a ``SparseNdarray`` to a NumPy array.

        Returns:
            Dense array of the same type as :py:attr:`~dtype` and shape as
            :py:attr:`~shape`.
        """
        indices = _spawn_indices(self._shape)
        return _extract_dense_array_from_SparseNdarray(self, indices)

    # Assorted dunder methods.
    def __add__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Add something to the right-hand-side of a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the addition.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="add", right=True)

    def __radd__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Add something to the left-hand-side of a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the addition.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="add", right=False)

    def __sub__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Subtract something from the right-hand-side of a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the subtraction.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="subtract", right=True)

    def __rsub__(self, other):
        """Subtract a ``SparseNdarray`` from something else.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the subtraction.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="subtract", right=False)

    def __mul__(self, other):
        """Multiply a ``SparseNdarray`` with something on the right hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the multiplication.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="multiply", right=True)

    def __rmul__(self, other):
        """Multiply a ``SparseNdarray`` with something on the left hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the multiplication.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="multiply", right=False)

    def __truediv__(self, other):
        """Divide a ``SparseNdarray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the division.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="divide", right=True)

    def __rtruediv__(self, other):
        """Divide something by a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the division.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="divide", right=False)

    def __mod__(self, other):
        """Take the remainder after dividing a ``SparseNdarray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the modulo.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="remainder", right=True)

    def __rmod__(self, other):
        """Take the remainder after dividing something by a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the modulo.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="remainder", right=False
        )

    def __floordiv__(self, other):
        """Divide a ``SparseNdarray`` by something and take the floor.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the floor division.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="floor_divide", right=True
        )

    def __rfloordiv__(self, other):
        """Divide something by a ``SparseNdarray`` and take the floor.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the floor division.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="floor_divide", right=False
        )

    def __pow__(self, other):
        """Raise a ``SparseNdarray`` to the power of something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the power operation.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="power", right=True)

    def __rpow__(self, other):
        """Raise something to the power of the contents of a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the power operation.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="power", right=False)

    def __eq__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check for equality between a ``SparseNdarray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="equal", right=True)

    def __req__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check for equality between something and a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="equal", right=False)

    def __ne__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check for non-equality between a ``SparseNdarray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="not_equal", right=True)

    def __rne__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check for non-equality between something and a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="not_equal", right=False)

    def __ge__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether a ``SparseNdarray`` is greater than or equal to something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="greater_equal", right=True)

    def __rge__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether something is greater than or equal to a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="greater_equal", right=False
        )

    def __le__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether a ``SparseNdarray`` is less than or equal to something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="less_equal", right=True
        )

    def __rle__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether something is greater than or equal to a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(
            self, other, operation="less_equal", right=False
        )

    def __gt__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether a ``SparseNdarray`` is greater than something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="greater", right=True)

    def __rgt__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether something is greater than a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="greater", right=False)

    def __lt__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether a ``SparseNdarray`` is less than something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="less", right=True)

    def __rlt__(self, other) -> Union["SparseNdarray", numpy.ndarray]:
        """Check whether something is less than a ``SparseNdarray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            Array containing the result of the check.
            This may or may not be sparse depending on ``other``.
        """
        return _operate_with_args_on_SparseNdarray(self, other, operation="less", right=False)

    # Simple methods.
    def __neg__(self):
        """Negate the contents of a ``SparseNdarray``.

        Returns:
            A ``SparseNdarray`` containing the delayed negation.
        """
        return _transform_sparse_array_from_SparseNdarray(self, lambda l, i, v : (i, -v), self._dtype)

    def __abs__(self):
        """Take the absolute value of the contents of a ``SparseNdarray``.

        Returns:
            A ``SparseNdarray`` containing the delayed absolute value operation.
        """
        return _transform_sparse_array_from_SparseNdarray(self, lambda l, i, v : (i, abs(v)), self._dtype)

    # Subsetting.
    def __getitem__(self, subset: Tuple[Union[slice, Sequence], ...]) -> Union["SparseNdarray", numpy.ndarray]:
        """Take a subset of this ``SparseNdarray``. This follows the same logic as NumPy slicing and will generate a
        :py:class:`~delayedarray.Subset.Subset` object when the subset operation preserves the dimensionality of the
        seed, i.e., ``args`` is defined using the :py:meth:`~numpy.ix_` function.

        Args:
            args:
                A :py:class:`tuple` of length equal to the dimensionality of this ``SparseNdarray``.
                Any NumPy slicing is supported but only subsets that preserve dimensionality will generate a
                delayed subset operation.

        Raises:
            ValueError: If ``args`` contain more dimensions than the shape of the array.

        Returns:
            If the dimensionality is preserved by ``args``, a ``SparseNdarray`` containing a delayed subset operation is
            returned. Otherwise, a :py:class:`~numpy.ndarray` is returned containing the realized subset.
        """
        cleaned = _getitem_subset_preserves_dimensions(self.shape, subset)
        if cleaned is not None:
            # No need to sanitize here, as the extractors can take unsorted subsets.
            return _extract_sparse_array_from_SparseNdarray(self, cleaned)
        return _getitem_subset_discards_dimensions(self, subset, _extract_dense_array_from_SparseNdarray)


    # NumPy methods.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "SparseNdarray":
        """Interface with NumPy array methods.  This is used to implement
        mathematical operations like NumPy's :py:meth:`~numpy.log`, or to
        override operations between NumPy class instances and ``SparseNdarray``
        objects where the former is on the left hand side. 

        Check out NumPy's ``__array_ufunc__`` `documentation
        <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__>`_
        for more details.

        Returns:
            A ``SparseNdarray`` instance containing the requested delayed operation.
        """
        if ufunc.__name__ in translate_ufunc_to_op_with_args or ufunc.__name__ == "true_divide":
            # This is required to support situations where the NumPy array is on
            # the LHS, such that the numpy.ndarray method gets called first.

            op = ufunc.__name__
            if ufunc.__name__ == "true_divide":
                op = "divide"

            first_is_da = isinstance(inputs[0], SparseNdarray)
            da = inputs[1 - int(first_is_da)]
            v = inputs[int(first_is_da)]
            return _operate_with_args_on_SparseNdarray(self, v, op, right=False)

        elif ufunc.__name__ in translate_ufunc_to_op_simple:
            dummy = ufunc(numpy.zeros(1, dtype=self._dtype))
            if dummy[0] == 0:
                return _transform_sparse_array_from_SparseNdarray(self, lambda l, i, v : (i, ufunc(v)), dummy.dtype)
            else:
                return ufunc(self.__array__())

        elif ufunc.__name__ == "absolute":
            return self.__abs__()

        raise NotImplementedError(f"'{ufunc.__name__}' is not implemented!")


    def __array_function__(self, func, types, args, kwargs) -> "SparseNdarray":
        """Interface to NumPy's high-level array functions.
        This is used to implement array operations like NumPy's :py:meth:`~numpy.concatenate`,

        Check out NumPy's ``__array_function__``
        `documentation <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__>`_
        for more details.

        Returns:
            A ``SparseNdarray`` instance containing the requested operation.
        """
        if func == numpy.concatenate:
            if "axis" in kwargs:
                axis = kwargs["axis"]
            else:
                axis = 0
            return _concatenate_SparseNdarrays(args[0], along=axis)

        if func == numpy.transpose:
            if "axes" in kwargs:
                axes = kwargs["axes"]
            else:
                axes = list(range(len(self._shape) - 1, -1, -1))
            return _transpose_SparseNdarray(self, axes)

        if func == numpy.round:
            return _transform_sparse_array_from_SparseNdarray(self, lambda l, i, v : (i, func(v, **kwargs)), self._dtype)

        raise NotImplementedError(f"'{func.__name__}' is not implemented!")


    def astype(self, dtype: numpy.dtype, **kwargs) -> "SparseNdarray":
        """See :py:meth:`~numpy.ndarray.astype` for details.

        All keyword arguments are currently ignored.
        """
        return _transform_sparse_array_from_SparseNdarray(self, lambda l, i, v : (i, v.astype(dtype)), dtype)


    @property
    def T(self) -> "SparseNdarray":
        """
        Returns:
            A ``SparseNdarray`` containing the transposed contents.
        """
        axes = list(range(len(self._shape) - 1, -1, -1))
        return _transpose_SparseNdarray(self, axes)


    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None) -> numpy.ndarray:
        """
        Take the sum of values across the ``SparseNdarray``, possibly over a
        given axis or set of axes.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.sum`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``SparseNdarray``, see
                ``numpy.sum`` for details.

        Returns:
            A NumPy array containing the summed values. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        return array_sum(
            self, 
            axis=axis, 
            dtype=dtype,
            reduce_over_x=_reduce_SparseNdarray,
            masked=self._is_masked,
        )


    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None) -> numpy.ndarray:
        """
        Take the mean of values across the ``SparseNdarray``, possibly over a
        given axis or set of axes.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.mean`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``SparseNdarray``, see
                ``numpy.mean`` for details.

        Returns:
            A NumPy array containing the mean values. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        return array_mean(
            self, 
            axis=axis, 
            dtype=dtype,
            reduce_over_x=_reduce_SparseNdarray,
            masked=self._is_masked,
        )


    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None, ddof: int = 0) -> numpy.ndarray:
        """
        Take the variances of values across the ``SparseNdarray``, possibly
        over a given axis or set of axes.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.mean`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``SparseNdarray``, see
                ``numpy.mean`` for details.

            ddof:
                Delta in the degrees of freedom to subtract from the denominator.
                Typically set to 1 to obtain the sample variance.

        Returns:
            A NumPy array containing the variances. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        return array_var(
            self, 
            axis=axis, 
            dtype=dtype,
            ddof=ddof,
            reduce_over_x=_reduce_SparseNdarray,
            masked=self._is_masked,
        )


    # Other stuff
    def __copy__(self) -> "SparseNdarray":
        """
        Returns:
            A deep copy of this object.
        """
        return SparseNdarray(
            shape=self._shape, 
            contents=copy.deepcopy(self._contents), 
            index_dtype=self._index_dtype, 
            dtype=self._dtype, 
            is_masked=self._is_masked, 
            check=False
        )


    def copy(self) -> "SparseNdarray":
        """
        Returns:
            A deep copy of this object.
        """
        return self.__copy__()


#########################################################
#########################################################


def _peek(contents: list, dim: int):
    if dim == 1:
        for x in contents:
            if x is not None:
                return x[0].dtype, x[1].dtype, numpy.ma.isMaskedArray(x[1])
    else:
        for x in contents:
            if x is not None:
                out = _peek(x, dim - 1)
                if out is not None:
                    return out
    return None


_CheckPayload = namedtuple("_CheckPayload", [ "max_index", "dtype", "index_dtype", "is_masked" ])


def _check_sparse_tuple(indices: numpy.ndarray, values: numpy.ndarray, payload: _CheckPayload):
    if len(indices) != len(values):
        raise ValueError("length of index and value vectors should be the same")

    if values.dtype != payload.dtype:
        raise ValueError("inconsistent data types for different value vectors")

    if indices.dtype != payload.index_dtype:
        raise ValueError("inconsistent data types for different index vectors")

    for i, ix in enumerate(indices):
        if ix < 0 or ix >= payload.max_index:
            raise ValueError("index vectors out of range for the last dimension")

    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            raise ValueError("index vectors should be sorted in strictly increasing order")

    if payload.is_masked != numpy.ma.isMaskedArray(values):
        raise ValueError("inconsistent masking status for different value vectors")


def _recursive_check(contents: list, shape: Tuple[int, ...], payload: _CheckPayload, dim: int):
    if len(contents) != shape[dim]:
        raise ValueError("length of 'contents' or its components should match the extent of the corresponding dimension")

    if dim == 1:
        for x in contents:
            if x is not None:
                _check_sparse_tuple(x[0], x[1], payload)
    else:
        for x in contents:
            if x is not None:
                _recursive_check(x, shape, payload, dim=dim-1)


#########################################################
#########################################################


_SubsetSummary = namedtuple("_SubsetSummary", [ "subset", "increasing", "consecutive", "search_first", "search_last", "first_index", "past_last_index", "random_map" ])


def _characterize_indices(subset: Sequence[int], dim_extent: int):
    if len(subset) == 0:
        return _SubsetSummary(
            subset = subset,
            increasing = False,
            consecutive = False,
            first_index = None,
            past_last_index = None,
            search_first = None,
            search_last = None,
            random_map = None,
        )

    for i in range(1, len(subset)):
        if subset[i] <= subset[i-1]:
            random_map = {}
            for i, x in enumerate(subset):
                if x in random_map:
                    if isinstance(random_map[x], list):
                        random_map[x].append(i)
                    else:
                        random_map[x] = [random_map[x], i]
                else:
                    random_map[x] = i
            first = min(subset)
            return _SubsetSummary(
                subset = subset, 
                increasing = False,
                consecutive = False,
                first_index = first,
                past_last_index = None,
                search_first = (first > 0),
                search_last = None,
                random_map = random_map
            )

    consecutive = True
    for i in range(1, len(subset)):
        if subset[i] != subset[i - 1] + 1:
            consecutive = False
            break

    first = subset[0]
    last = subset[-1] + 1
    return _SubsetSummary(
        subset = subset, 
        increasing = True,
        consecutive = consecutive,
        first_index = first,
        past_last_index = last,
        search_first = (first > 0), 
        search_last = (last < dim_extent), 
        random_map = None,
    )


def _extract_sparse_vector_internal(indices: numpy.ndarray, values: numpy.ndarray, subset_summary: _SubsetSummary, f: Callable):
    subset = subset_summary.subset

    start_pos = 0
    if subset_summary.search_first:
        start_pos = bisect_left(indices, subset_summary.first_index)
    end_pos = len(indices)

    if subset_summary.increasing:
        pos = 0
        x = start_pos
        for i in subset:
            while x < end_pos and i > indices[x]:
                x += 1
            if x == end_pos:
                break
            if i == indices[x]:
                f(pos, i, values[x])
            pos += 1
    else: 
        for x in range(start_pos, end_pos):
            ix = indices[x]
            if ix in subset_summary.random_map:
                targets = subset_summary.random_map[ix]
                if isinstance(targets, int):
                    f(targets, ix, values[x])
                else:
                    for t in targets:
                        f(t, ix, values[x])


def _extract_sparse_vector_to_dense(indices: numpy.ndarray, values: numpy.ndarray, subset_summary: _SubsetSummary, output: numpy.ndarray):
    if len(subset_summary.subset) == 0:
        pass
    elif subset_summary.consecutive:
        start_pos = 0
        first = subset_summary.first_index
        if subset_summary.search_first:
            start_pos = bisect_left(indices, first)

        end_pos = len(indices)
        if subset_summary.search_last:
            end_pos = bisect_left(indices, subset_summary.past_last_index, lo=start_pos, hi=end_pos)

        for x in range(start_pos, end_pos):
            output[indices[x] - first] = values[x]
    else:
        def f(p, i, v):
            output[p] = v
        _extract_sparse_vector_internal(indices, values, subset_summary, f)


def _recursive_extract_dense_array(contents: numpy.ndarray, subset: Tuple[Sequence[int], ...], subset_summary: _SubsetSummary, output: numpy.ndarray, dim: int):
    curdex = subset[dim]
    if dim == 1:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _extract_sparse_vector_to_dense(x[0], x[1], subset_summary=subset_summary, output=output[pos])
            pos += 1
    else:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _recursive_extract_dense_array(
                    contents=x, 
                    subset=subset, 
                    subset_summary=subset_summary,
                    output=output[pos],
                    dim=dim - 1, 
                )
            pos += 1


def _extract_dense_array_from_SparseNdarray(x: SparseNdarray, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    idims = [len(y) for y in subset]
    subset_summary = _characterize_indices(subset[0], x._shape[0])

    # We reverse the dimensions so that we use F-contiguous storage. This also
    # makes it slightly easier to do the recursion as we can just index by
    # the first dimension to obtain a subarray at each recursive step. 
    output = numpy.zeros((*reversed(idims),), dtype=x._dtype)

    if x._contents is not None:
        if x._is_masked:
            output = numpy.ma.MaskedArray(output, mask=False)
        ndim = len(x._shape)
        if ndim > 1:
            _recursive_extract_dense_array(x._contents, subset, subset_summary=subset_summary, output=output, dim=ndim-1)
        else:
            _extract_sparse_vector_to_dense(x._contents[0], x._contents[1], subset_summary=subset_summary, output=output)

    return output.T


def _extract_sparse_vector_to_sparse(indices: numpy.ndarray, values: numpy.ndarray, subset_summary: _SubsetSummary):
    if len(subset_summary.subset) == 0:
        pass

    elif subset_summary.consecutive:
        start_pos = 0
        first = subset_summary.first_index
        if subset_summary.search_first:
            start_pos = bisect_left(indices, first)

        end_pos = len(indices)
        if subset_summary.search_last:
            end_pos = bisect_left(indices, subset_summary.past_last_index, lo=start_pos, hi=end_pos)

        if start_pos == 0 and end_pos == len(indices):
            new_indices = indices
            new_values = values
        else:
            new_indices = indices[start_pos:end_pos]
            new_values = values[start_pos:end_pos]

        if first:
            new_indices = new_indices - first
        return new_indices, new_values

    elif subset_summary.increasing:
        new_indices = []
        new_values = []
        def f(p, i, v):
            new_indices.append(p)
            new_values.append(v)
        _extract_sparse_vector_internal(indices, values, subset_summary, f)

        if len(new_indices) == 0:
            return None
        return (
            _convert_to_unmasked_1darray(new_indices, dtype=indices.dtype), 
            _convert_to_maybe_masked_1darray(new_values, dtype=values.dtype, masked=numpy.ma.isMaskedArray(values))
        )

    else:
        new_pairs = []
        def f(p, i, v):
            new_pairs.append((p, v))
        _extract_sparse_vector_internal(indices, values, subset_summary, f)
        new_pairs.sort()

        shape = (len(new_pairs),)
        new_indices = _allocate_unmasked_ndarray(shape, dtype=indices.dtype)
        new_values = _allocate_maybe_masked_ndarray(shape, dtype=values.dtype, masked=numpy.ma.isMaskedArray(values))
        for i, pair in enumerate(new_pairs):
            new_indices[i] = pair[0]
            new_values[i] = pair[1]

        return new_indices, new_values


def _recursive_extract_sparse_array(contents: list, shape: Tuple[int, ...], subset: Tuple[Sequence[int], ...], subset_summary: _SubsetSummary, dim: int):
    curdex = subset[dim]
    new_contents = []

    if dim == 1:
        for i in curdex:
            x = contents[i]
            if x is not None:
                y = _extract_sparse_vector_to_sparse(x[0], x[1], subset_summary)
                new_contents.append(y)
            else:
                new_contents.append(None)
    else:
        for i in curdex:
            if contents[i] is not None:
                y = _recursive_extract_sparse_array(
                    contents[i], 
                    shape, 
                    subset=subset, 
                    subset_summary=subset_summary,
                    dim=dim - 1, 
                )
                new_contents.append(y)
            else:
                new_contents.append(None)

    for x in new_contents:
        if x is not None:
            return new_contents
    return None


def _extract_sparse_array_from_SparseNdarray(x: SparseNdarray, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    idims = [len(y) for y in subset]
    subset_summary = _characterize_indices(subset[0], x._shape[0])

    new_contents = None
    if x._contents is not None:
        ndim = len(x.shape)
        if ndim > 1:
            new_contents = _recursive_extract_sparse_array(x._contents, x._shape, subset=subset, subset_summary=subset_summary, dim=ndim-1)
        else:
            new_contents = _extract_sparse_vector_to_sparse(x._contents[0], x._contents[1], subset_summary)

    return SparseNdarray(shape=(*idims,), contents=new_contents, index_dtype=x.index_dtype, dtype=x.dtype, is_masked=x.is_masked, check=False)


#########################################################
#########################################################


_TransformPayload = namedtuple("_TransformPayload", [ "fun", "output_dtype" ])


def _transform_sparse_vector(location: Sequence[int], indices: numpy.ndarray, values: numpy.ndarray, payload: _TransformPayload):
    idx, val = payload.fun(location, indices, values)
    return (idx.astype(indices.dtype, copy=False), val.astype(payload.output_dtype, copy=False)) # a bit of safety with respect to types.


def _recursive_transform_sparse_array(contents: list, shape: Tuple[int, ...], payload: _TransformPayload, dim: int, location: Sequence[int] = []):
    new_contents = []
    location.append(0)

    if dim == 1:
        for i in range(shape[dim]):
            location[-1] = i
            x = contents[i]
            if x is not None:
                new_contents.append(_transform_sparse_vector(location, x[0], x[1], payload))
            else:
                new_contents.append(None)
    else:
        for i in range(shape[dim]):
            if contents[i] is not None:
                location[-1] = i
                y = _recursive_transform_sparse_array(
                    contents=contents[i], 
                    shape=shape, 
                    payload=payload,
                    dim=dim-1,
                    location=location, 
                )
                new_contents.append(y)
            else:
                new_contents.append(None)

    location.pop()

    for x in new_contents:
        if x is not None:
            return new_contents
    return None


def _transform_sparse_array_from_SparseNdarray(x: SparseNdarray, f: Callable, output_dtype: numpy.dtype) -> SparseNdarray:
    new_contents = None
    if x._contents is not None:
        payload = _TransformPayload(fun=f, output_dtype=output_dtype)
        ndim = len(x._shape)
        if ndim > 1:
            new_contents = _recursive_transform_sparse_array(contents=x._contents, shape=x._shape, payload=payload, dim=ndim-1)
        else:
            new_contents = _transform_sparse_vector((), indices=x._contents[0], values=x._contents[1], payload=payload)

    return SparseNdarray(shape=x._shape, contents=new_contents, index_dtype=x.index_dtype, dtype=output_dtype, is_masked=x.is_masked, check=False)


#########################################################
#########################################################


_BinaryOpPayload = namedtuple("_BinaryOpPayload", [ "fun", "dtype1", "dtype2", "output_dtype", "output_index_dtype" ])


def _binary_operate_sparse_vector(vector1: Tuple[numpy.ndarray, numpy.ndarray], vector2: Tuple[numpy.ndarray, numpy.ndarray], payload: _BinaryOpPayload):
    if vector1 is None and vector2 is None:
        return None

    elif vector1 is not None and vector2 is None:
        indices1, values1 = vector1
        mock = numpy.zeros((1,), dtype=payload.dtype2) # get vector of length 1 for correct type coercion.
        return indices1.astype(payload.output_index_dtype, copy=False), payload.fun(values1, mock).astype(payload.output_dtype, copy=False)

    elif vector1 is None and vector2 is not None:
        indices2, values2 = vector2
        mock = numpy.zeros((1,), dtype=payload.dtype1)
        return indices2.astype(payload.output_index_dtype, copy=False), payload.fun(mock, values2).astype(payload.output_dtype, copy=False)

    else:
        indices1, values1 = vector1
        indices2, values2 = vector2
        f = payload.fun

        i1 = 0
        len1 = len(indices1)
        z1 = values1.dtype.type(0)
        i2 = 0
        len2 = len(indices2)
        z2 = values2.dtype.type(0)
        outval = []
        outidx = []

        while i1 < len1 and i2 < len2:
            ix1 = indices1[i1]
            ix2 = indices2[i2]
            if ix1 > ix2:
                outval.append(f(z1, values2[i2]))
                outidx.append(ix2)
                i2 += 1
            elif ix1 < ix2:
                outval.append(f(values1[i1], z2))
                outidx.append(ix1)
                i1 += 1
            else:
                outval.append(f(values1[i1], values2[i2]))
                outidx.append(ix1)
                i1 += 1
                i2 += 1

        # Only one of the following should be called.
        while i2 < len2:
            outval.append(f(z1, values2[i2]))
            outidx.append(indices2[i2])
            i2 += 1

        while i1 < len1:
            outval.append(f(values1[i1], z2))
            outidx.append(indices1[i1])
            i1 += 1

        return (
            _convert_to_unmasked_1darray(outidx, dtype=payload.output_index_dtype), 
            _convert_to_maybe_masked_1darray(outval, dtype=payload.output_dtype, masked=numpy.ma.isMaskedArray(values1) or numpy.ma.isMaskedArray(values2))
        )


def _recursive_binary_operation_on_SparseNdarray(contents1: list, contents2: list, payload: _BinaryOpPayload, dim: int):
    if contents1 is None and contents2 is None:
        return None

    new_contents = []
    if contents1 is not None and contents2 is None:
        if dim == 1:
            for con1 in contents1:
                new_contents.append(_binary_operate_sparse_vector(con1, None, payload))
        else:
            for con1 in contents1:
                new_contents.append(_recursive_binary_operation_on_SparseNdarray(con1, None, payload, dim=dim - 1))

    elif contents1 is None and contents2 is not None:
        if dim == 1:
            for con2 in contents2:
                new_contents.append(_binary_operate_sparse_vector(None, con2, payload))
        else:
            for con2 in contents2:
                new_contents.append(_recursive_binary_operation_on_SparseNdarray(None, con2, payload, dim=dim - 1))

    else:
        if dim == 1:
            for i, con1 in enumerate(contents1):
                new_contents.append(_binary_operate_sparse_vector(con1, contents2[i], payload))
        else:
            for i, con1 in enumerate(contents1):
                new_contents.append(_recursive_binary_operation_on_SparseNdarray(con1, contents2[i], payload, dim=dim - 1))

    for x in new_contents:
        if x is not None:
            return new_contents
    return None


def _binary_operation_on_SparseNdarray(x: SparseNdarray, y: SparseNdarray, operation: ISOMETRIC_OP_WITH_ARGS):
    op = _choose_operator(operation)

    dummy1 = numpy.zeros(1, dtype=x._dtype)
    dummy2 = numpy.zeros(1, dtype=y._dtype)
    dummy = op(dummy1, dummy2)
    if dummy[0] != 0:
        return op(x.__array__(), y.__array__())

    if x._contents is None and y._contents is None:
        new_contents = None
    else:
        payload = _BinaryOpPayload(fun=op, dtype1=x._dtype, dtype2=y._dtype, output_index_dtype=x.index_dtype, output_dtype=dummy.dtype)
        ndim = len(x._shape)
        if ndim > 1:
            new_contents = _recursive_binary_operation_on_SparseNdarray(x._contents, y._contents, payload=payload, dim = ndim - 1)
        else:
            new_contents = _binary_operate_sparse_vector(x._contents, y._contents, payload=payload)

    return SparseNdarray(shape=x._shape, contents=new_contents, index_dtype=x.index_dtype, dtype=dummy.dtype, is_masked=x.is_masked, check=False)


#########################################################
#########################################################


def _operate_with_args_on_SparseNdarray(x: SparseNdarray, other, operation: ISOMETRIC_OP_WITH_ARGS, right: bool) -> SparseNdarray:
    if isinstance(other, SparseNdarray):
        return _binary_operation_on_SparseNdarray(x, other, operation)

    along = _infer_along_with_args(x._shape, other)
    num_other = 1

    op = _choose_operator(operation)
    dummy = numpy.zeros(num_other, dtype=x._dtype)
    if right:
        dummy = op(dummy, other)
    else:
        dummy = op(other, dummy)

    if num_other and not (dummy == 0).all(): # densifying.
        if right:
            return op(x.__array__(), other)
        else:
            return op(other, x.__array__())

    if isinstance(other, numpy.ndarray):
        num_other = numpy.prod(other.shape)
        other = other.reshape((num_other,)) # flattening

    if along is None:
        if right:
            def f2(location, indices, values):
                return indices, op(values, other)
        else:
            def f2(location, indices, values):
                return indices, op(other, values)
    elif along > 0:
        if right:
            def f2(location, indices, values):
                p = location[-along] # remember, location is (i) reversed and (ii) missing the final dimension, so '-along' works.
                return indices, op(values, other[p:p+1]) # get vector of length 1 for correct type coercion.
        else:
            def f2(location, indices, values):
                p = location[-along]
                return indices, op(other[p:p+1], values)
    else:
        if right:
            def f2(location, indices, values):
                return indices, op(values, other[indices])
        else:
            def f2(location, indices, values):
                return indices, op(other[indices], values)

    return _transform_sparse_array_from_SparseNdarray(x, f2, dummy.dtype) 


#########################################################
#########################################################


_TransposeFillPayload = namedtuple("_TransposeFillPayload", [ "perm", "new_shape", "new_contents" ])


def _transpose_SparseNdarray_contents_internal(location: Sequence[int], indices: numpy.ndarray, values: numpy.ndarray, payload: _TransposeFillPayload):
    destination = []
    final = None
    for i, p in enumerate(payload.perm):
        if p == 0:
            final = i
            destination.append(None)
        else:
            destination.append(location[-p]) # remember, location is (i) reversed and (ii) missing the final dimension, so '-p' works.

    ndim = len(payload.new_shape)
    for i, ix in enumerate(indices):
        destination[final] = ix

        target = payload.new_contents
        for j in range(ndim - 1, 1, -1): # walking backwards from the later dimensions to fill 'new_contents' correctly.
            d = destination[j]
            if target[d] is None:
                replacement = [None] * payload.new_shape[j - 1]
                target[d] = replacement 
            target = target[d]

        d = destination[1] 
        if target[d] is None:
            target[d] = ([], [])
        outi, outv = target[d]
        outi.append(destination[0])
        outv.append(values[i])


def _recursive_transpose_SparseNdarray_fill(contents: list, payload: _TransposeFillPayload, dim: int, location: Sequence[int] = []):
    location.append(0)

    if dim == 1:
        for i, con in enumerate(contents):
            if con is not None:
                location[-1] = i
                _transpose_SparseNdarray_contents_internal(location, con[0], con[1], payload)
    else:
        for i, con in enumerate(contents):
            if con is not None:
                location[-1] = i
                _recursive_transpose_SparseNdarray_fill(con, payload, location=location, dim=dim - 1)

    location.pop()


_TransposeReallocPayload = namedtuple("_TransposeReallocatePayload", [ "output_dtype", "output_index_dtype", "output_is_masked" ])


def _recursive_transpose_SparseNdarray_reallocate(contents: list, payload: _TransposeReallocPayload, dim: int):
    if dim == 1:
        for i, con in enumerate(contents):
            if con is not None:
                contents[i] = (
                    _convert_to_unmasked_1darray(con[0], dtype=payload.output_index_dtype), 
                    _convert_to_maybe_masked_1darray(con[1], dtype=payload.output_dtype, masked=payload.output_is_masked)
                )
    else:
        for i, con in enumerate(contents):
            if con is not None:
                _recursive_transpose_SparseNdarray_reallocate(con, payload, dim=dim - 1)


def _transpose_SparseNdarray(x: SparseNdarray, perm):
    ndim = len(x._shape)
    if ndim == 1:
        return x

    new_shape = []
    for p in perm:
        new_shape.append(x._shape[p])

    new_contents = None
    if x._contents is not None:
        new_contents = [None] * new_shape[-1]

        _recursive_transpose_SparseNdarray_fill(
            x._contents, 
            _TransposeFillPayload(perm=perm, new_shape=new_shape, new_contents=new_contents),
            dim=ndim - 1,
        )

        _recursive_transpose_SparseNdarray_reallocate(
            new_contents, 
            _TransposeReallocPayload(output_dtype=x._dtype, output_index_dtype=x._index_dtype, output_is_masked=x._is_masked),
            dim=ndim - 1,
        )

    return SparseNdarray(shape=(*new_shape,), contents=new_contents, index_dtype=x._index_dtype, dtype=x._dtype, is_masked=x.is_masked, check=False)


#########################################################
#########################################################


_ConcatenatePayload = namedtuple("_ConcatenatePayload", [ "shapes", "offset", "output_dtype", "output_index_dtype", "output_is_masked"])


def _concatenate_sparse_vectors(idx: List[numpy.ndarray], val: List[numpy.ndarray], payload: _ConcatenatePayload):
    newidx = _concatenate_unmasked_ndarrays(idx, axis=0).astype(payload.output_index_dtype, copy=False)
    newval = _concatenate_maybe_masked_ndarrays(val, axis=0, masked=payload.output_is_masked).astype(payload.output_dtype, copy=False)
    return (newidx, newval)


def _coerce_concatenated_SparseNdarray_types(contents: list, payload: _ConcatenatePayload, dim: int):
    if dim == 1:
        for i, con in enumerate(contents):
            if con is not None:
                idx2 = con[0].astype(payload.output_index_dtype, copy=False)
                val2 = con[1].astype(payload.output_dtype, copy=False)
                contents[i] = (idx2, val2)
    else:
        for i, con in enumerate(contents):
            if con is not None:
                _coerce_concatenated_SparseNdarray_types(con, payload, dim=dim - 1)


def _recursive_concatenate_SparseNdarrays(contents: list, final_shape: Tuple[int, ...], along: int, payload: _ConcatenatePayload, dim: int):
    if along == dim:
        all_none = True
        for x in contents:
            if x is not None:
                all_none = False

        new_contents = None
        if not all_none:
            new_contents = []
            for i, x in enumerate(contents):
                if x is not None:
                    new_contents += x
                else:
                    new_contents += [None] * payload.shapes[i][along]
            _coerce_concatenated_SparseNdarray_types(new_contents, payload=payload, dim=dim)
        return new_contents

    elif dim == 1:
        new_contents = []
        for i in range(final_shape[dim]):
            outidx = []
            outval = [] 
            for j, c in enumerate(contents):
                if c is not None and c[i] is not None:
                    outidx.append(c[i][0] + payload.offset[j])
                    outval.append(c[i][1])
            if len(outval):
                new_contents.append(_concatenate_sparse_vectors(outidx, outval, payload))
            else:
                new_contents.append(None)
        return new_contents

    else:
        new_contents = []
        collected = [None] * len(contents)
        for i in range(final_shape[dim]):
            for j, c in enumerate(contents):
                if c is not None:
                    collected[j] = c[i]
            new_contents.append(_recursive_concatenate_SparseNdarrays(collected, final_shape, along, payload, dim=dim-1))
        return new_contents


def _concatenate_SparseNdarrays(xs: List[SparseNdarray], along: int):
    all_contents = []
    all_shapes = []
    for x in xs:
        all_contents.append(x._contents)
        all_shapes.append(x._shape)

    combined = 0
    ref_shape = all_shapes[0]
    for shape in all_shapes:
        if len(shape) != len(ref_shape):
            raise ValueError("inconsistent dimensionalities for combining SparseNdarrays")
        for d, s in enumerate(shape):
            if d == along:
                combined += s
            elif s != ref_shape[d]:
                raise ValueError("inconsistent shapes for combining SparseNdarrays along axis " + str(along))

    new_shape = list(ref_shape)
    new_shape[along] = combined

    dummy_collected = []
    dummy_collected_index = []
    for x in xs:
        dummy_collected.append(numpy.zeros(1, dtype=x._dtype))
        dummy_collected_index.append(numpy.zeros(1, dtype=x._index_dtype))

    output_dtype = _concatenate_unmasked_ndarrays(dummy_collected, axis=0).dtype
    output_index_dtype = _concatenate_unmasked_ndarrays(dummy_collected_index, axis=0).dtype
    output_is_masked = any(y._is_masked for y in xs)

    all_none = True
    for con in all_contents:
        if con is not None:
            all_none = False

    new_contents = None
    if not all_none:
        offset = None
        ndim = len(new_shape)

        # Upgrading the index type to ensure we can hold the output, if we're
        # concatenating on the first dimension (and thus possibly increasing
        # the indices in the leaves of the subsequent SparseNdarrays).
        if along == 0:
            last = 0
            offset = []
            for i, shape in enumerate(all_shapes):
                offset.append(last)
                last += shape[along]

            for candidate in [output_index_dtype, numpy.uint32, numpy.uint64]:
                if last < numpy.iinfo(candidate).max:
                    output_index_dtype = candidate
                    break

        payload = _ConcatenatePayload(
            shapes=all_shapes, 
            offset=offset, 
            output_dtype=output_dtype, 
            output_index_dtype=output_index_dtype, 
            output_is_masked=output_is_masked
        )

        if ndim > 1:
            new_contents = _recursive_concatenate_SparseNdarrays(all_contents, new_shape, along=along, payload=payload, dim=ndim-1)
        else:
            outidx = []
            outval = [] 
            for j, c in enumerate(all_contents):
                if c is not None:
                    outidx.append(c[0] + offset[j])
                    outval.append(c[1])
            new_contents = _concatenate_sparse_vectors(outidx, outval, payload)

    return SparseNdarray(shape=(*new_shape,), contents=new_contents, dtype=output_dtype, index_dtype=output_index_dtype, is_masked=output_is_masked, check=False)


#########################################################
#########################################################


_ReductionPayload = namedtuple("_ReductionPayload", [ "multipliers", "operation" ])


def _reduce_sparse_vector(idx: numpy.ndarray, val: numpy.ndarray, payload: _ReductionPayload, offset: int = 0):
    for j, ix in enumerate(idx):
        extra = payload.multipliers[0] * ix
        payload.operation(offset + extra, val[j])
    return


def _recursive_reduce_SparseNdarray(contents, payload: _ReductionPayload, dim: int, offset: int  = 0):
    mult = payload.multipliers[dim]
    if dim == 1:
        for i, con in enumerate(contents):
            if con is not None:
                _reduce_sparse_vector(con[0], con[1], payload, offset = offset + mult * i)
    else:
        for i, con in enumerate(contents):
            if con is not None:
                _recursive_reduce_SparseNdarray(con, payload, dim = dim - 1, offset = offset + mult * i)
    return


def _reduce_SparseNdarray(x: SparseNdarray, axes: List[int], operation: Callable):
    if x.contents is not None:
        multipliers = _create_offset_multipliers(x.shape, axes)
        payload = _ReductionPayload(operation=operation, multipliers=multipliers)
        ndim = len(x.shape)
        if ndim == 1:
            _reduce_sparse_vector(x.contents[0], x.contents[1], payload)
        else:
            _recursive_reduce_SparseNdarray(x.contents, payload, dim=ndim - 1)
    return        
