import numbers
from bisect import bisect_left
from typing import Callable, List, Optional, Sequence, Tuple, Union
from collections import namedtuple
from numpy import array, ndarray, zeros, dtype

from ._getitem import _sanitize_getitem, _extract_dense_subarray

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class SparseNdarray:
    """The SparseNdarray, as its name suggests, is a sparse n-dimensional array.
    It is inspired by the **SVTArray** from the `DelayedArray R/Bioconductor package <https://bioconductor.org/packages/DelayedArray>`_.
    This class is primarily intended for developers, either as a seed to newly constructed
    (sparse) :py:class:`~delayedarray.DelayedArray.DelayedArray` instances or
    as the output of :py:meth:`~delayedarray.interface.extract_sparse_array`;
    end-users should be interacting with :py:class:`~delayedarray.DelayedArray.DelayedArray` instances instead.

    Internally, the SparseNdarray is represented as a nested list where each
    nesting level corresponds to a dimension. The list at each level has length equal
    to the extent of its dimension, where each entry contains another list representing
    the contents of the corresponding element of that dimension. This proceeds until
    the penultimate dimension, where each entry instead ``(index, value)`` tuples.
    In effect, this is a tree where the non-leaf nodes are lists and the leaf nodes
    are tuples.

    Each ``(index, value)`` tuple represents a sparse vector for the corresponding element of the final dimension of the SparseNdarray.
    ``index`` should be a :py:class:`~typing.Sequence` of integers where
    entries are strictly increasing and less than the extent of the final dimension.
    ``value`` may be any :py:class:`~numpy.ndarray` but the ``dtype`` should be
    consistent across all ``value`` objects in the SparseNdarray.

    Any entry of any list may also be None, indicating that the corresponding element
    of the dimension contains no non-zero values. In fact, the entire tree may be None,
    indicating that there are no non-zero values in the entire array.

    For 1-dimensional arrays, the contents should be a single ``(index, value)`` tuple
    containing the sparse contents. This may also be None if there are no non-zero
    values in the array.

    For 0-dimensional arrays, the contents should be a single numeric scalar, or None.

    Attributes:
        shape (Tuple[int, ...]):
            Tuple specifying the dimensions of the array.

        contents (Union[Tuple[Sequence, Sequence], List], optional):
            For ``n``-dimensional arrays where ``n`` > 1, a nested list representing a
            tree where each leaf node is a tuple containing a sparse vector (or None).

            For 1-dimensional arrays, a tuple containing a sparse vector.

            Alternatively None, if the array is empty.

        dtype (dtype, optional):
            NumPy type of the SparseNdarray.
            If None, this is inferred from ``contents``.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        contents: Optional[
            Union[
                Tuple[Sequence, Sequence],
                List,
            ]
        ],
        dtype: Optional[dtype] = None,
        check=True,
    ):
        self._shape = shape
        self._contents = contents

        if dtype is None:
            if contents is not None:
                if len(shape) > 1:
                    dtype = _peek_for_type(contents, 0, self._shape)
                elif len(shape) == 1:
                    dtype = contents[1].dtype
            if dtype is None:
                raise ValueError("cannot infer 'dtype' from 'contents'")
        self._dtype = dtype

        if check is True and contents is not None:
            if len(shape) > 1:
                _recursive_check(self._contents, 0, self._shape, self._dtype)
            elif len(shape) == 1:
                _check_sparse_tuple(
                    self._contents[0], self._contents[1], self._shape[0], self._dtype
                )
            else:
                if not isinstance(contents, numbers.Number):
                    raise ValueError(
                        "expected a numeric scalar 'contents' for a 0-dimensional SparseNdarray"
                    )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the SparseNdarray.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the SparseNdarray.
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Type of the elements in the SparseNdarray.

        Returns:
            dtype: NumPy type of the values.
        """
        return self._dtype

    @property
    def contents(self):
        """Contents of the array. This is intended to be read-only; in general, ``contents`` should only be modified by
        developers of :py:meth:`~delayedarray.interface.extract_sparse_array` methods or creators of new
        :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.

        Returns:
            A nested list, for a n-dimensional array where n > 1.

            A tuple containing a sparse vector (i.e., indices and values), for a 1-dimensional array.

            A single scalar, for a 0-dimensional array.

            Alternatively None, if the array contains no non-zero elements.
        """
        return self._contents


    def __repr__(self) -> str:
        """Pretty-print this ``SparseNdarray``. This uses :py:meth:`~numpy.array2string` and responds to all of its
        options.

        Returns:
            str: String containing a prettified display of the array contents.
        """
        total = 1
        for s in self._seed.shape:
            total *= s

        preamble = "<" + " x ".join([str(x) for x in self._shape]) + ">"
        preamble += " " + type(self).__name__ + " object of type '" + self._dtype.name + "'"

        if total > get_printoptions()["threshold"]:
            ndims = len(self._seed.shape)
            indices = []
            edge_size = get_printoptions()["edgeitems"]
            for d in range(ndims):
                extent = self._seed.shape[d]
                if extent > edge_size * 2:
                    indices.append(
                        list(range(edge_size + 1))
                        + list(range(extent - edge_size, extent))
                    )
                else:
                    indices.append(slice(None))
            indices = (*indices,)
        else:
            indices = [range(d) for d in self._shape]

        bits_and_pieces = _extract_dense_array_from_SparseNdarray(self, indices)
        converted = array2string(bits_and_pieces, separator=", ", threshold=0)
        return preamble + "\n" + converted


    # For NumPy:
    def __array__(self) -> ndarray:
        """Convert a ``SparseNdarray`` to a NumPy array.

        Returns:
            ndarray: Array of the same type as :py:attr:`~dtype` and shape as :py:attr:`~shape`.
            This is guaranteed to be in C-contiguous order and to not be a view on other data.
        """
        indices = [range(d) for d in self._shape]
        return _extract_dense_array_from_SparseNdarray(self, indices)


    def astype(self, dtype, **kwargs):
        """See :py:meth:`~numpy.ndarray.astype` for details.

        All keyword arguments are currently ignored.
        """
        return _transform_sparse_array_from_SparseNdarray(self)


    # Assorted dunder methods.
    def __add__(self, other) -> "DelayedArray":
        """Add something to the right-hand-side of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed addition operation.
        """
        return _wrap_isometric_with_args(self, other, operation="add", right=True)

    def __radd__(self, other) -> "DelayedArray":
        """Add something to the left-hand-side of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed addition operation.
        """
        return _wrap_isometric_with_args(self, other, operation="add", right=False)

    def __sub__(self, other) -> "DelayedArray":
        """Subtract something from the right-hand-side of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed subtraction operation.
        """
        return _wrap_isometric_with_args(self, other, operation="subtract", right=True)

    def __rsub__(self, other):
        """Subtract a ``DelayedArray`` from something else.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed subtraction operation.
        """
        return _wrap_isometric_with_args(self, other, operation="subtract", right=False)

    def __mul__(self, other):
        """Multiply a ``DelayedArray`` with something on the right hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed multiplication operation.
        """
        return _wrap_isometric_with_args(self, other, operation="multiply", right=True)

    def __rmul__(self, other):
        """Multiply a ``DelayedArray`` with something on the left hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed multiplication operation.
        """
        return _wrap_isometric_with_args(self, other, operation="multiply", right=False)

    def __truediv__(self, other):
        """Divide a ``DelayedArray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed division operation.
        """
        return _wrap_isometric_with_args(self, other, operation="divide", right=True)

    def __rtruediv__(self, other):
        """Divide something by a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed division operation.
        """
        return _wrap_isometric_with_args(self, other, operation="divide", right=False)

    def __mod__(self, other):
        """Take the remainder after dividing a ``DelayedArray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed modulo operation.
        """
        return _wrap_isometric_with_args(self, other, operation="remainder", right=True)

    def __rmod__(self, other):
        """Take the remainder after dividing something by a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed modulo operation.
        """
        return _wrap_isometric_with_args(
            self, other, operation="remainder", right=False
        )

    def __floordiv__(self, other):
        """Divide a ``DelayedArray`` by something and take the floor.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed floor division operation.
        """
        return _wrap_isometric_with_args(
            self, other, operation="floor_divide", right=True
        )

    def __rfloordiv__(self, other):
        """Divide something by a ``DelayedArray`` and take the floor.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed floor division operation.
        """
        return _wrap_isometric_with_args(
            self, other, operation="floor_divide", right=False
        )

    def __pow__(self, other):
        """Raise a ``DelayedArray`` to the power of something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed power operation.
        """
        return _wrap_isometric_with_args(self, other, operation="power", right=True)

    def __rpow__(self, other):
        """Raise something to the power of the contents of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed power operation.
        """
        return _wrap_isometric_with_args(self, other, operation="power", right=False)

    def __eq__(self, other) -> "DelayedArray":
        """Check for equality between a ``DelayedArray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="equal", right=True)

    def __req__(self, other) -> "DelayedArray":
        """Check for equality between something and a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="equal", right=False)

    def __ne__(self, other) -> "DelayedArray":
        """Check for non-equality between a ``DelayedArray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="not_equal", right=True)

    def __rne__(self, other) -> "DelayedArray":
        """Check for non-equality between something and a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(
            self, other, operation="not_equal", right=False
        )

    def __ge__(self, other) -> "DelayedArray":
        """Check whether a ``DelayedArray`` is greater than or equal to something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(
            self, other, operation="greater_equal", right=True
        )

    def __rge__(self, other) -> "DelayedArray":
        """Check whether something is greater than or equal to a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(
            self, other, operation="greater_equal", right=False
        )

    def __le__(self, other) -> "DelayedArray":
        """Check whether a ``DelayedArray`` is less than or equal to something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(
            self, other, operation="less_equal", right=True
        )

    def __rle__(self, other) -> "DelayedArray":
        """Check whether something is greater than or equal to a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(
            self, other, operation="less_equal", right=False
        )

    def __gt__(self, other) -> "DelayedArray":
        """Check whether a ``DelayedArray`` is greater than something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="greater", right=True)

    def __rgt__(self, other) -> "DelayedArray":
        """Check whether something is greater than a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="greater", right=False)

    def __lt__(self, other) -> "DelayedArray":
        """Check whether a ``DelayedArray`` is less than something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="less", right=True)

    def __rlt__(self, other) -> "DelayedArray":
        """Check whether something is less than a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or any seed object of the same dimensions as :py:attr:`~shape`.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="less", right=False)

    # Simple methods.
    def __neg__(self):
        """Negate the contents of a ``DelayedArray``.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed negation.
        """
        return _wrap_isometric_with_args(self, 0, operation="subtract", right=False)

    def __abs__(self):
        """Take the absolute value of the contents of a ``DelayedArray``.

        Returns:
            DelayedArray: A ``DelayedArray`` containing the delayed absolute value operation.
        """
        return DelayedArray(UnaryIsometricOpSimple(self._seed, operation="abs"))

    # Subsetting.
    def __getitem__(
        self, args: Tuple[Union[slice, Sequence[Union[int, bool]]], ...]
    ) -> Union["DelayedArray", ndarray]:
        """Take a subset of this ``DelayedArray``. This follows the same logic as NumPy slicing and will generate a
        :py:class:`~delayedarray.Subset.Subset` object when the subset operation preserves the dimensionality of the
        seed, i.e., ``args`` is defined using the :py:meth:`~numpy.ix_` function.

        Args:
            args (Tuple[Union[slice, Sequence[Union[int, bool]]], ...]):
                A :py:class:`tuple` of length equal to the dimensionality of this ``DelayedArray``.
                Any NumPy slicing is supported but only subsets that preserve dimensionality will generate a
                delayed subset operation.

        Raises:
            ValueError: If ``args`` contain more dimensions than the shape of the array.

        Returns:
            If the dimensionality is preserved by ``args``, a ``DelayedArray`` containing a delayed subset operation is
            returned. Otherwise, a :py:class:`~numpy.ndarray` is returned containing the realized subset.
        """
        sanitized = _sanitize_getitem(self._shape, args)
        if sanitized is not None:
            return _extract_sparse_array_from_SparseNdarray(self, sanitized)
        return _extract_dense_subarray(self, self._shape, args)


    # Coercion methods.
    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        return self.__array__()


    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        return _extract_sparse_array_from_SparseNdarray(self, subset)


    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        total = [1] * len(self._shape)
        total[-1] = self._shape[-1]
        return (*total,)


    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return True



#########################################################
#########################################################


def _peek_for_type(contents: Sequence, dim: int, shape: Tuple[int, ...]):
    ndim = len(shape)
    if dim == ndim - 2:
        for x in contents:
            if x is not None:
                return x[1].dtype
    else:
        for x in contents:
            if x is not None:
                out = _peek_for_type(x, dim + 1, shape)
                if out is not None:
                    return out
    return None


def _check_sparse_tuple(
    indices: Sequence, values: ndarray, max_index: int, dtype: dtype
):
    if len(indices) != len(values):
        raise ValueError("Length of index and value vectors should be the same.")

    if values.dtype != dtype:
        raise ValueError("Inconsistent data types for different value vectors.")

    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= max_index:
            raise ValueError("Index vectors out of range for the last dimension.")

    for i in range(1, len(indices)):
        if indices[i] <= indices[i - 1]:
            raise ValueError("Index vectors should be sorted.")


def _recursive_check(
    contents: Sequence, dim: int, shape: Tuple[int, ...], dtype: dtype
):
    if len(contents) != shape[dim]:
        raise ValueError(
            "Length of 'contents' or its components should match the extent of the corresponding dimension."
        )

    ndim = len(shape)
    if dim == ndim - 2:
        for x in contents:
            if x is not None:
                _check_sparse_tuple(x[0], x[1], shape[ndim - 1], dtype)
    else:
        for x in contents:
            if x is not None:
                _recursive_check(x, dim + 1, shape, dtype)


#########################################################
#########################################################


_SubsetSummary = namedtuple("_SubsetSummary", [ "subset", "consecutive", "search_first", "search_last", "first_index", "past_last_index" ])


def _characterize_indices(subset: Sequence, dim: int):
    if len(subset) == 0:
        return _SubsetSummary(
            subset=subset, 
            consecutive=False, 
            search_first=False, 
            search_last=False, 
            first_index=None, 
            past_last_index=None
        )

    first = subset[0]
    last = subset[-1] + 1
    consecutive = True
    for i in range(1, len(subset)):
        if subset[i] != subset[i - 1] + 1:
            consecutive = False
            break

    return _SubsetSummary(
        subset=subset, 
        consecutive=consecutive, 
        search_first=(first > 0), 
        search_last=(last < dim), 
        first_index=first,
        past_last_index=last,
    )


def _extract_sparse_vector_internal(
    indices: Sequence,
    values: Sequence,
    subset_summary: _SubsetSummary,
    f: Callable,
):
    subset = subset_summary.subset
    if len(subset) == 0:
        return

    start_pos = 0
    if subset_summary.search_first:
        start_pos = bisect_left(indices, subset_summary.first_index)

    if subset_summary.consecutive:
        end_pos = len(indices)
        if subset_summary.search_last:
            end_pos = bisect_left(indices, subset_summary.past_last_index, lo=start_pos, hi=end_pos)
        first = subset_summary.first_index
        for x in range(start_pos, end_pos):
            f(indices[x] - first, indices[x], values[x])
    else:
        pos = 0
        x = start_pos
        xlen = len(indices)
        for i in subset:
            while x < xlen and i > indices[x]:
                x += 1
            if x == xlen:
                break
            if i == indices[x]:
                f(pos, i, values[x])
            pos += 1


def _extract_sparse_vector_to_dense(indices, values, subset_summary, output):
    def f(p, i, v):
        output[p] = v
    _extract_sparse_vector_internal(indices, values, subset_summary, f)


def _recursive_extract_dense_array(contents, ndim, subset, dim, output):
    curdex = subset[dim]
    if dim == ndim - 2:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _extract_sparse_vector_to_dense(x[0], x[1], subset[ndim - 1], output[pos])
            pos += 1
    else:
        pos = 0
        for i in curdex:
            x = contents[i]
            if x is not None:
                _recursive_extract_dense_array(x, ndim, subset, dim + 1, output[pos])
            pos += 1


def _extract_dense_array_from_SparseNdarray(x: SparseNdarray, subset: Tuple[Union[slice, Sequence], ...]) -> ndarray:
    subset2 = list(subset)
    idims = [len(y) for y in subset2]
    subset2[-1] = _characterize_indices(subset2[-1], x._shape[-1])

    output = zeros((*idims,), dtype=x._dtype)
    if x._contents is not None:
        ndims = len(x.shape)
        if ndims > 1:
            _recursive_extract_dense_array(x._contents, ndims, subset2, 0, output)
        else:
            _extract_sparse_vector_to_dense(x._contents[0], x._contents[1], subset2[0], output)

    return output


def _extract_sparse_vector_to_sparse(indices, values, subset_summary):
    new_indices = []
    new_values = []

    def f(p, i, v):
        new_indices.append(p)
        new_values.append(v)
    _extract_sparse_vector_internal(indices, values, subset_summary, f)

    if len(new_indices) == 0:
        return None
    if isinstance(indices, ndarray):
        new_indices = array(new_indices, dtype=indices.dtype)
    if isinstance(values, ndarray):
        new_values = array(new_values, dtype=values.dtype)
    return new_indices, new_values


def _recursive_extract_sparse_array(contents, shape, subset, dim):
    ndim = len(shape)
    curdex = subset[dim]
    new_contents = []

    if dim == ndim - 2:
        pos = 0
        last_subset = subset[ndim - 1]
        for i in curdex:
            x = contents[i]
            if x is not None:
                y = _extract_sparse_vector_to_sparse(x[0], x[1], last_subset)
                new_contents.append(y)
            else:
                new_contents.append(None)
            pos += 1
    else:
        pos = 0
        for i in curdex:
            if contents[i] is not None:
                y = _recursive_extract_sparse_array(contents[i], shape, subset, dim + 1)
                new_contents.append(y)
            else:
                new_contents.append(None)
            pos += 1

    for x in new_contents:
        if x is not None:
            return new_contents
    return None


def _extract_sparse_array_from_SparseNdarray(x: SparseNdarray, subset: Tuple[Union[slice, Sequence], ...]) -> SparseNdarray:
    subset2 = list(subset)
    idims = [len(y) for y in subset2]
    subset2[-1] = _characterize_indices(subset2[-1], x._shape[-1])

    new_contents = None
    if x._contents is not None:
        if len(x.shape) > 1:
            new_contents = _recursive_extract_sparse_array(x._contents, x._shape, subset2, 0)
        else:
            new_contents = _extract_sparse_vector_to_sparse(x._contents[0], x._contents[1], subset2[0])

    return SparseNdarray(shape=(*idims,), contents=new_contents, dtype=x.dtype, check=False)
