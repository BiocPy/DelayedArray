from typing import Sequence, Tuple, Union, TYPE_CHECKING

import numpy
from numpy import (
    array,
    array2string,
    dtype,
    get_printoptions,
    integer,
    issubdtype,
    ndarray,
    prod,
)

if TYPE_CHECKING:
    import dask.array

import warnings
from .BinaryIsometricOp import BinaryIsometricOp
from .Cast import Cast
from .Combine import Combine
from .Round import Round
from .Subset import Subset
from .Transpose import Transpose
from .UnaryIsometricOpSimple import UnaryIsometricOpSimple
from .UnaryIsometricOpWithArgs import UnaryIsometricOpWithArgs
from .utils import create_dask_array, extract_array, _densify, chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def _wrap_isometric_with_args(x, other, operation, right):
    if hasattr(other, "shape") and other.shape == x.shape:
        if right:
            left = x
            right = other
        else:
            left = other
            right = x
        return DelayedArray(
            BinaryIsometricOp(_extract_seed(left), _extract_seed(right), operation)
        )

    return DelayedArray(
        UnaryIsometricOpWithArgs(
            _extract_seed(x),
            value=other,
            operation=operation,
            right=right,
        )
    )


def _extract_seed(x):
    if isinstance(x, DelayedArray):
        return x._seed
    else:
        return x


translate_ufunc_to_op_with_args = set(
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "remainder",
        "floor_divide",
        "power",
        "equal",
        "greater_equal",
        "greater",
        "less_equal",
        "less",
        "not_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
    ]
)

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

    This is equivalent to the class of the same name from
    the `R/Bioconductor package <https://bioconductor.org/packages/DelayedArray>`_ of the same name.
    It allows users to efficiently operate on large matrices without actually evaluating the
    operation or creating new copies; instead, the operations will transparently return another ``DelayedArray``
    instance containing the delayed operations, which can be realized by calling :py:meth:`~numpy.array` or related
    methods.

    Attributes:
        seed: Any array-like object that satisfies the seed contract.
            This means that it has the :py:attr:`~shape` and :py:attr:`~dtype` properties.

            In addition, it should either have an :py:meth:`~__DelayedArray_extract__` method, or
            it should suppoort NumPy slicing via :py:meth:`~numpy.ix_`. Additional NumPy
            interface support (e.g., dunder methods, ufuncs) will be used where relevant.

            For dask support, the seed should provide a :py:meth:`~__DelayedArray_dask__`
            method if it is not already compatible with dask.
    """

    def __init__(self, seed):
        self._seed = seed

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``DelayedArray``.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``DelayedArray``.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the elements in the ``DelayedArray``.

        Returns:
            dtype: NumPy type of the values.
        """
        return self._seed.dtype

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    @property
    def T(self) -> "DelayedArray":
        """Get the delayed transpose of this ``DelayedArray`` instance.

        Returns:
            DelayedArray: A DelayedArray containing a delayed transposition.
        """
        return DelayedArray(Transpose(self._seed, perm=None))

    def __repr__(self) -> str:
        """Pretty-print this ``DelayedArray``. This uses :py:meth:`~numpy.array2string` and responds to all of its
        options.

        Returns:
            str: String containing a prettified display of the array contents.
        """
        total = 1
        for s in self._seed.shape:
            total *= s

        preamble = "<" + " x ".join([str(x) for x in self._seed.shape]) + ">"
        preamble += " DelayedArray object of type '" + self._seed.dtype.name + "'"

        indices = None
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

        bits_and_pieces = _densify(extract_array(self._seed, indices))
        converted = array2string(bits_and_pieces, separator=", ", threshold=0)
        return preamble + "\n" + converted

    # For NumPy:
    def __array__(self) -> ndarray:
        """Convert a ``DelayedArray`` to a NumPy array.

        Returns:
            ndarray: Array of the same type as :py:attr:`~dtype` and shape as :py:attr:`~shape`.
            This is guaranteed to be in C-contiguous order and to not be a view on other data.
        """
        return _densify(extract_array(self._seed))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "DelayedArray":
        """Interface with NumPy array methods.

        This is used to implement mathematical operations like NumPy's :py:meth:`~numpy.log`,
        or to override operations between NumPy class instances and ``DelayedArray`` objects where the former is on the
        left hand side. Check out the NumPy's ``__array_ufunc__``
        `documentation <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__>`_ for
        more details.

        Returns:
            DelayedArray: A ``DelayedArray`` instance containing the requested delayed operation.
        """
        if (
            ufunc.__name__ in translate_ufunc_to_op_with_args
            or ufunc.__name__ == "true_divide"
        ):
            # This is required to support situations where the NumPy array is on
            # the LHS, such that the ndarray method gets called first.

            op = ufunc.__name__
            if ufunc.__name__ == "true_divide":
                op = "divide"

            first_is_da = isinstance(inputs[0], DelayedArray)
            da = inputs[1 - int(first_is_da)]
            v = inputs[int(first_is_da)]
            return _wrap_isometric_with_args(
                _extract_seed(da), v, operation=op, right=first_is_da
            )
        elif ufunc.__name__ in translate_ufunc_to_op_simple:
            return DelayedArray(
                UnaryIsometricOpSimple(
                    _extract_seed(inputs[0]), operation=ufunc.__name__
                )
            )
        elif ufunc.__name__ == "absolute":
            return DelayedArray(
                UnaryIsometricOpSimple(_extract_seed(inputs[0]), operation="abs")
            )

        raise NotImplementedError(f"'{ufunc.__name__}' is not implemented!")

    def __array_function__(self, func, types, args, kwargs):
        """Interface to NumPy's high-level array functions.
        This is used to implement array operations like NumPy's :py:meth:`~numpy.concatenate`,
        Check out the NumPy's ``__array_function__``
        `documentation <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__>`_
        for more details.

        Returns:
            DelayedArray: A ``DelayedArray`` instance containing the requested delayed operation.
        """
        if func == numpy.concatenate:
            seeds = []
            for x in args[0]:
                seeds.append(_extract_seed(x))

            if "axis" in kwargs:
                axis = kwargs["axis"]
            else:
                axis = 0
            return DelayedArray(Combine(seeds, along=axis))

        if func == numpy.transpose:
            seed = _extract_seed(args[0])
            if "axes" in kwargs:
                axes = kwargs["axes"]
            else:
                axes = None
            return DelayedArray(Transpose(seed, perm=axes))

        if func == numpy.round:
            seed = _extract_seed(args[0])
            if "decimals" in kwargs:
                decimals = kwargs["decimals"]
            else:
                decimals = 0
            return DelayedArray(Round(seed, decimals=decimals))

        raise NotImplementedError(f"'{func.__name__}' is not implemented!")

    def astype(self, dtype, **kwargs):
        """See :py:meth:`~numpy.ndarray.astype` for details.

        All keyword arguments are currently ignored.
        """
        return DelayedArray(Cast(self._seed, dtype))

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

        ndim = len(self.shape)
        if not isinstance(args, tuple):
            args = [args] + [slice(None)] * (ndim - 1)
        if len(args) < ndim:
            args = list(args) + [slice(None)] * (ndim - len(args))
        elif len(args) > ndim:
            raise ValueError(
                "more indices in 'args' than there are dimensions in 'seed'"
            )

        # Checking if we're preserving the shape via a cross index.
        cross_index = True
        for d, idx in enumerate(args):
            if (
                not isinstance(idx, ndarray)
                or not issubdtype(idx.dtype, integer)
                or len(idx.shape) != ndim
            ):
                cross_index = False
                break

            for d2 in range(ndim):
                if d != d2 and idx.shape[d2] != 1:
                    cross_index = False
                    break

        if cross_index:
            sanitized = []
            for d, idx in enumerate(args):
                sanitized.append(idx.reshape((prod(idx.shape),)))
            return DelayedArray(Subset(self._seed, (*sanitized,)))

        # Checking if we're preserving the shape via a slice.
        slices = 0
        failed = False
        for d, idx in enumerate(args):
            if isinstance(idx, slice):
                slices += 1
                continue
            elif isinstance(idx, ndarray):
                if len(idx.shape) != 1:
                    failed = True
                    break
            elif not isinstance(idx, Sequence):
                failed = True
                break

        if not failed and slices >= ndim - 1:
            sanitized = []
            for d, idx in enumerate(args):
                if isinstance(idx, slice):
                    sanitized.append(range(*idx.indices(self.shape[d])))
                else:
                    dummy = array(range(self.shape[d]))[idx]
                    sanitized.append(dummy)
            return DelayedArray(Subset(self._seed, (*sanitized,)))

        # If we're discarding dimensions, we see if we can do some pre-emptive extraction.
        failed = False
        as_vector = []
        new_args = []
        dim_loss = 0

        for d, idx in enumerate(args):
            if isinstance(idx, ndarray):
                if len(idx.shape) != 1:
                    failed = True
                    break
            elif isinstance(idx, slice):
                idx = range(*idx.indices(self.shape[d]))
            elif not isinstance(idx, Sequence):
                as_vector.append([idx])
                new_args.append(0)
                dim_loss += 1
                continue

            as_vector.append(idx)
            new_args.append(slice(None))

        if not failed:
            # Just using Subset here to avoid having to reproduce the
            # uniquifying/sorting of subsets before extract_array().
            base_seed = extract_array(Subset(self._seed, (*as_vector,)))
        else:
            base_seed = extract_array(self._seed)
            new_args = args

        try:
            test = base_seed[(..., *new_args)]
            if len(test.shape) != ndim - dim_loss:
                raise ValueError(
                    "slicing for "
                    + str(type(base_seed))
                    + " does not discard dimensions with scalar indices"
                )
        except Exception as e:
            warnings.warn(str(e))
            test = _densify(base_seed)[(..., *new_args)]

        if len(test.shape) == ndim:
            raise NotImplementedError(
                "Oops. Looks like the DelayedArray doesn't correctly handle this combination of index types, but it "
                "probably should. Consider filing an issue in at https://github.com/BiocPy/DelayedArray/issues."
            )

        return test

    # For python-level compute.
    def sum(self, *args, **kwargs):
        """See :py:meth:`~numpy.sums` for details."""
        target = extract_array(self._seed)
        try:
            return target.sum(*args, **kwargs)
        except Exception as e:
            warnings.warn(str(e))
            target = _densify(target)
            return target.sum(*args, **kwargs)

    def var(self, *args, **kwargs):
        """See :py:meth:`~numpy.vars` for details."""
        target = extract_array(self._seed)
        try:
            return target.var(*args, **kwargs)
        except Exception as e:
            warnings.warn(e)
            target = _densify(target)
            return target.var(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """See :py:meth:`~numpy.means` for details."""
        target = extract_array(self._seed)
        try:
            return target.mean(*args, **kwargs)
        except Exception as e:
            warnings.warn(e)
            target = _densify(target)
            return target.mean(*args, **kwargs)

    # Coercion methods.
    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        return create_dask_array(self._seed)

    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        return extract_array(self._seed, subset)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self._seed)
