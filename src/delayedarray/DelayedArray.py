from typing import Sequence, Tuple, Union, Optional, List, Callable
import numpy
from numpy import array, dtype, integer, issubdtype, ndarray, prod, array2string
from collections import namedtuple

from .SparseNdarray import SparseNdarray
from .BinaryIsometricOp import BinaryIsometricOp
from .Cast import Cast
from .Combine import Combine
from .Round import Round
from .Subset import Subset
from .Transpose import Transpose
from .UnaryIsometricOpSimple import UnaryIsometricOpSimple
from .UnaryIsometricOpWithArgs import UnaryIsometricOpWithArgs

from .extract_dense_array import extract_dense_array
from .to_dense_array import to_dense_array
from .extract_sparse_array import extract_sparse_array
from .apply_over_blocks import apply_over_blocks
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

from ._subset import _getitem_subset_preserves_dimensions, _getitem_subset_discards_dimensions, _repr_subset
from ._isometric import translate_ufunc_to_op_simple, translate_ufunc_to_op_with_args
from ._statistics import array_mean, array_var, array_sum, _create_offset_multipliers

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


class DelayedArray:
    """Array containing delayed operations. This is equivalent to the class of
    the same name from the `R/Bioconductor package
    <https://bioconductor.org/packages/DelayedArray>`_ of the same name.  It
    allows users to efficiently operate on large matrices without actually
    evaluating the operation or creating new copies; instead, the operations
    will transparently return another ``DelayedArray`` instance containing the
    delayed operations, which can be realized by calling
    :py:meth:`~numpy.array` or related methods.

    Any object that satisfies the "seed contract" can be wrapped by a
    ``DelayedArray``. Specifically, a seed should have:

    - The :py:attr:`~shape` and :py:attr:`~dtype` properties, which are of the
      same type as the corresponding properties of NumPy arrays.
    - A method for the
      :py:meth:`~delayedarray.extract_dense_array.extract_dense_array` generic.
    - A method for the :py:meth:`~delayedarray.is_masked.is_masked` generic.

    If the seed contains sparse data, it should also implement:

    - A method for the :py:meth:`~delayedarray.is_sparse.is_sparse` generic.
    - A method for the
      :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`
      generic.

    Optionally, a seed class may have:

    - A method for the :py:meth:`~delayedarray.chunk_shape.chunk_shape` generic,
      if there is some preferred dimension in which to take chunks of the array.
    - A method for the
      :py:meth:`~delayedarray.create_dask_array.create_dask_array` generic,
      if the seed is not already compatible with the **dask** package.
    """

    def __init__(self, seed):
        """Most users should use :py:meth:`~delayedarray.wrap.wrap`
        instead, as this can be specialized by developers to construct
        subclasses that are optimized for custom seed types.

        Args:
            seed: Any array-like object that satisfies the seed contract.
        """
        self._seed = seed

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of the ``DelayedArray``.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """
        Returns:
            NumPy type of the elements in the ``DelayedArray``.
        """
        return self._seed.dtype

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed

    @property
    def T(self) -> "DelayedArray":
        """
        Returns:
            A ``DelayedArray`` containing the delayed transpose.
        """
        return DelayedArray(Transpose(self._seed, perm=None))

    def __repr__(self) -> str:
        """Pretty-print this ``DelayedArray``. This uses
        :py:meth:`~numpy.array2string` and responds to all of its options.

        Returns:
            String containing a prettified display of the array contents.
        """
        preamble = "<" + " x ".join([str(x) for x in self._seed.shape]) + ">"
        if is_sparse(self._seed):
            preamble += " sparse"
        preamble += " " + type(self).__name__ + " object of type '" + self._seed.dtype.name + "'"

        indices = _repr_subset(self._seed.shape)
        bits_and_pieces = extract_dense_array(self._seed, indices)
        converted = array2string(bits_and_pieces, separator=", ", threshold=0)
        return preamble + "\n" + converted

    # For NumPy:
    def __array__(self) -> ndarray:
        """Convert a ``DelayedArray`` to a NumPy array, to be used by
        :py:meth:`~numpy.array`. 

        Returns:
            NumPy array of the same type as :py:attr:`~dtype` and shape as
            :py:attr:`~shape`. 
        """
        return to_dense_array(self._seed)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> "DelayedArray":
        """Interface with NumPy array methods. This is used to implement
        mathematical operations like NumPy's :py:meth:`~numpy.log`, or to
        override operations between NumPy class instances and ``DelayedArray``
        objects where the former is on the left hand side. 

        Check out NumPy's ``__array_ufunc__`` `documentation
        <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__>`_
        for more details.

        Returns:
            A ``DelayedArray`` instance containing the requested delayed operation.
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
        elif ufunc.__name__ == "logical_not":
            return DelayedArray(
                UnaryIsometricOpSimple(_extract_seed(inputs[0]), operation="logical_not")
            )

        raise NotImplementedError(f"'{ufunc.__name__}' is not implemented!")

    # Just get the array priority above that of the numpy MaskedArray so that
    # we call DelayedArray's __array_ufunc__ override instead... annoyingly, it
    # doesn't actually work (https://github.com/numpy/numpy/issues/15200).
    __array_priority__ = numpy.ma.MaskedArray.__array_priority__ + 1

    def __array_function__(self, func, types, args, kwargs) -> "DelayedArray":
        """Interface to NumPy's high-level array functions.  This is used to
        implement array operations like NumPy's :py:meth:`~numpy.concatenate`,

        Check out NumPy's ``__array_function__`` `documentation
        <https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__>`_
        for more details.

        Returns:
            A ``DelayedArray`` instance containing the requested delayed operation.
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

        if func == numpy.shape:
            return self.shape 

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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed addition operation.
        """
        return _wrap_isometric_with_args(self, other, operation="add", right=True)

    def __radd__(self, other) -> "DelayedArray":
        """Add something to the left-hand-side of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed addition operation.
        """
        return _wrap_isometric_with_args(self, other, operation="add", right=False)

    def __sub__(self, other) -> "DelayedArray":
        """Subtract something from the right-hand-side of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed subtraction operation.
        """
        return _wrap_isometric_with_args(self, other, operation="subtract", right=True)

    def __rsub__(self, other):
        """Subtract a ``DelayedArray`` from something else.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed subtraction operation.
        """
        return _wrap_isometric_with_args(self, other, operation="subtract", right=False)

    def __mul__(self, other):
        """Multiply a ``DelayedArray`` with something on the right hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed multiplication operation.
        """
        return _wrap_isometric_with_args(self, other, operation="multiply", right=True)

    def __rmul__(self, other):
        """Multiply a ``DelayedArray`` with something on the left hand side.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed multiplication operation.
        """
        return _wrap_isometric_with_args(self, other, operation="multiply", right=False)

    def __truediv__(self, other):
        """Divide a ``DelayedArray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed division operation.
        """
        return _wrap_isometric_with_args(self, other, operation="divide", right=True)

    def __rtruediv__(self, other):
        """Divide something by a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed division operation.
        """
        return _wrap_isometric_with_args(self, other, operation="divide", right=False)

    def __mod__(self, other):
        """Take the remainder after dividing a ``DelayedArray`` by something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` object of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed modulo operation.
        """
        return _wrap_isometric_with_args(self, other, operation="remainder", right=True)

    def __rmod__(self, other):
        """Take the remainder after dividing something by a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed modulo operation.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed floor division operation.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed floor division operation.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed power operation.
        """
        return _wrap_isometric_with_args(self, other, operation="power", right=True)

    def __rpow__(self, other):
        """Raise something to the power of the contents of a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed power operation.
        """
        return _wrap_isometric_with_args(self, other, operation="power", right=False)

    def __eq__(self, other) -> "DelayedArray":
        """Check for equality between a ``DelayedArray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="equal", right=True)

    def __req__(self, other) -> "DelayedArray":
        """Check for equality between something and a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="equal", right=False)

    def __ne__(self, other) -> "DelayedArray":
        """Check for non-equality between a ``DelayedArray`` and something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="not_equal", right=True)

    def __rne__(self, other) -> "DelayedArray":
        """Check for non-equality between something and a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
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
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="greater", right=True)

    def __rgt__(self, other) -> "DelayedArray":
        """Check whether something is greater than a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="greater", right=False)

    def __lt__(self, other) -> "DelayedArray":
        """Check whether a ``DelayedArray`` is less than something.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="less", right=True)

    def __rlt__(self, other) -> "DelayedArray":
        """Check whether something is less than a ``DelayedArray``.

        Args:
            other:
                A numeric scalar;
                or a NumPy array with dimensions as described in
                :py:class:`~delayedarray.UnaryIsometricOpWithArgs.UnaryIsometricOpWithArgs`;
                or a ``DelayedArray`` of the same dimensions as :py:attr:`~shape`.

        Returns:
            A ``DelayedArray`` containing the delayed check.
        """
        return _wrap_isometric_with_args(self, other, operation="less", right=False)

    # Simple methods.
    def __neg__(self) -> "DelayedArray":
        """Negate the contents of a ``DelayedArray``.

        Returns:
            A ``DelayedArray`` containing the delayed negation.
        """
        return _wrap_isometric_with_args(self, 0, operation="subtract", right=False)

    def __abs__(self) -> "DelayedArray":
        """Take the absolute value of the contents of a ``DelayedArray``.

        Returns:
            A ``DelayedArray`` containing the delayed absolute value operation.
        """
        return DelayedArray(UnaryIsometricOpSimple(self._seed, operation="abs"))

    # Subsetting.
    def __getitem__(self, subset: Tuple[Union[slice, Sequence], ...]) -> Union["DelayedArray", ndarray]:
        """Take a subset of this ``DelayedArray``. This follows the same logic as NumPy slicing and will generate a
        :py:class:`~delayedarray.Subset.Subset` object when the subset operation preserves the dimensionality of the
        seed, i.e., ``args`` is defined using the :py:meth:`~numpy.ix_` function.

        Args:
            subset:
                A :py:class:`tuple` of length equal to the dimensionality of
                this ``DelayedArray``.  We attempt to support most types of
                NumPy slicing; however, only subsets that preserve
                dimensionality will generate a delayed subset operation.

        Returns:
            If the dimensionality is preserved by ``subset``, a
            ``DelayedArray`` containing a delayed subset operation is returned.
            Otherwise, a :py:class:`~numpy.ndarray` is returned containing the
            realized subset.
        """
        cleaned = _getitem_subset_preserves_dimensions(self.shape, subset)
        if cleaned is not None:
            return DelayedArray(Subset(self._seed, cleaned))
        return _getitem_subset_discards_dimensions(self._seed, subset, extract_dense_array)


    # For python-level compute.
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None, buffer_size: int = 1e8) -> numpy.ndarray:
        """
        Take the sum of values across the ``DelayedArray``, possibly over a
        given axis or set of axes. If the seed has a ``sum()`` method, that
        method is called directly with the supplied arguments.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.sum`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``DelayedArray``, see
                ``numpy.sum`` for details.

            buffer_size:
                Buffer size in bytes to use for block processing. Larger values
                generally improve speed at the cost of memory.

        Returns:
            A NumPy array containing the summed values. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        if hasattr(self._seed, "sum"):
            return self._seed.sum(axis=axis, dtype=dtype)
        else:
            return array_sum(
                self, 
                axis=axis, 
                dtype=dtype, 
                reduce_over_x=lambda x, axes, op : _reduce(x, axes, op, buffer_size),
                masked=is_masked(self),
            )


    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None, buffer_size: int = 1e8) -> numpy.ndarray:
        """
        Take the mean of values across the ``DelayedArray``, possibly over a
        given axis or set of axes. If the seed has a ``mean()`` method, that
        method is called directly with the supplied arguments.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.mean`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``DelayedArray``, see
                ``numpy.mean`` for details.

            buffer_size:
                Buffer size in bytes to use for block processing. Larger values
                generally improve speed at the cost of memory.

        Returns:
            A NumPy array containing the mean values. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        if hasattr(self._seed, "mean"):
            return self._seed.mean(axis=axis, dtype=dtype)
        else:
            return array_mean(
                self, 
                axis=axis, 
                dtype=dtype, 
                reduce_over_x=lambda x, axes, op : _reduce(x, axes, op, buffer_size),
                masked=is_masked(self),
            )


    def var(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[numpy.dtype] = None, ddof: int = 0, buffer_size: int = 1e8) -> numpy.ndarray:
        """
        Take the variances of values across the ``DelayedArray``, possibly over
        a given axis or set of axes. If the seed has a ``var()`` method, that
        method is called directly with the supplied arguments.

        Args:
            axis: 
                A single integer specifying the axis to perform the calculation.
                Alternatively a tuple or None, see ``numpy.var`` for details.

            dtype:
                NumPy type for the output array. If None, this is automatically
                chosen based on the type of the ``DelayedArray``, see
                ``numpy.var`` for details.

            ddof:
                Delta in the degrees of freedom to subtract from the denominator.
                Typically set to 1 to obtain the sample variance.

            buffer_size:
                Buffer size in bytes to use for block processing. Larger values
                generally improve speed at the cost of memory.

        Returns:
            A NumPy array containing the variances. If ``axis = None``,
            this will be a NumPy scalar instead.
        """
        if hasattr(self._seed, "var"):
            return self._seed.var(axis=axis, dtype=dtype, ddof=ddof)
        else:
            return array_var(
                self, 
                axis=axis, 
                dtype=dtype, 
                ddof=ddof,
                reduce_over_x=lambda x, axes, op : _reduce(x, axes, op, buffer_size),
                masked=is_masked(self),
            )


@extract_dense_array.register
def extract_dense_array_DelayedArray(x: DelayedArray, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return extract_dense_array(x._seed, subset)


@extract_sparse_array.register
def extract_sparse_array_DelayedArray(x: DelayedArray, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return extract_sparse_array(x._seed, subset)


@create_dask_array.register
def create_dask_array_DelayedArray(x: DelayedArray):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    return create_dask_array(x._seed)


@chunk_shape.register
def chunk_shape_DelayedArray(x: DelayedArray):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return chunk_shape(x._seed)


@is_sparse.register
def is_sparse_DelayedArray(x: DelayedArray):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return is_sparse(x._seed)


@is_masked.register
def is_masked_DelayedArray(x: DelayedArray):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed)


#########################################################
#########################################################


_StatisticsPayload = namedtuple("_StatisticsPayload", [ "operation", "multipliers", "starts" ])


def _reduce_1darray(val: numpy.ndarray, payload: _StatisticsPayload, offset: int = 0):
    for i, v in enumerate(val):
        extra = payload.multipliers[-1] * (i + payload.starts[-1])
        payload.operation(offset + extra, v)
    return


def _recursive_reduce_ndarray(x: numpy.ndarray, payload: _StatisticsPayload, dim: int, offset: int = 0):
    mult = payload.multipliers[dim]
    shift = payload.starts[dim]
    if len(x.shape) == 2:
        for i in range(x.shape[0]):
            _reduce_1darray(x[i], payload, offset = offset + mult * (shift + i))
    else:
        for i in range(x.shape[0]):
            _recursive_reduce_ndarray(x[i], payload, dim = dim + 1, offset = offset + mult * (shift + i))
    return


def _reduce_ndarray(block: numpy.ndarray, multipliers: List[int], axes: List[int], position: Tuple[Tuple[int, int], ...], operation: Callable):
    ndim = len(block.shape)
    payload = _StatisticsPayload(operation=operation, multipliers=multipliers, starts=(*(s[0] for s in position),))
    if ndim == 1:
        _reduce_1darray(block, payload)
    else:
        _recursive_reduce_ndarray(block, payload, dim=0)
    return        


def _reduce_sparse_vector(idx: numpy.ndarray, val: numpy.ndarray, payload: _StatisticsPayload, offset: int = 0):
    for j, ix in enumerate(idx):
        extra = payload.multipliers[0] * (ix + payload.starts[0])
        payload.operation(offset + extra, val[j])
    return


def _recursive_reduce_SparseNdarray(contents, payload: _StatisticsPayload, dim: int, offset: int  = 0):
    mult = payload.multipliers[dim]
    start = payload.starts[dim]
    if dim == 1:
        for i, con in enumerate(contents):
            if con is not None:
                _reduce_sparse_vector(con[0], con[1], payload, offset = offset + mult * (i + start))
    else:
        for i, con in enumerate(contents):
            if con is not None:
                _recursive_reduce_SparseNdarray(con, payload, dim = dim - 1, offset = offset + mult * (i + start))
    return


def _reduce_SparseNdarray(x: SparseNdarray, multipliers: List[int], axes: List[int], position: Tuple[Tuple[int, int], ...], operation: Callable):
    if x.contents is not None:
        payload = _StatisticsPayload(operation=operation, multipliers=multipliers, starts=(*(s[0] for s in position),))
        ndim = len(x.shape)
        if ndim == 1:
            _reduce_sparse_vector(x.contents[0], x.contents[1], payload)
        else:
            _recursive_reduce_SparseNdarray(x.contents, payload, dim=ndim - 1)
    return        


def _reduce(x: DelayedArray, axes: List[int], operation: Callable, buffer_size: int):
    multipliers = _create_offset_multipliers(x.shape, axes)
    if is_sparse(x):
        apply_over_blocks(
            x, 
            lambda position, block : _reduce_SparseNdarray(block, multipliers, axes, position, operation), 
            buffer_size=buffer_size,
            allow_sparse=True,
        )
    else:
        apply_over_blocks(
            x, 
            lambda position, block : _reduce_ndarray(block, multipliers, axes, position, operation), 
            buffer_size=buffer_size,
        )
    return
