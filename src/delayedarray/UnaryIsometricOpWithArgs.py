import warnings
from typing import Literal, Tuple, Union, Sequence, TYPE_CHECKING

import numpy
from numpy import ndarray

if TYPE_CHECKING:
    import dask.array

from .utils import create_dask_array, extract_array, _retry_single, chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

OP = Literal[
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


def _execute(left, right, operation):
    # Can't use match/case yet, as that's only in Python 3.10, and we can't
    # just dispatch to 'getattr(numpy, operation)', because some classes don't
    # implement __array_func__. Thanks a lot, scipy.sparse, and fuck you.
    if operation == "add":
        return left + right
    elif operation == "subtract":
        return left - right
    elif operation == "multiply":
        return left * right
    elif operation == "divide":
        return left / right
    elif operation == "remainder":
        return left % right
    elif operation == "floor_divide":
        return left // right
    elif operation == "power":
        return left**right
    elif operation == "equal":
        return left == right
    elif operation == "greater_equal":
        return left >= right
    elif operation == "greater":
        return left > right
    elif operation == "less_equal":
        return left <= right
    elif operation == "less":
        return left < right
    elif operation == "not_equal":
        return left != right
    return getattr(numpy, operation)(left, right)


class UnaryIsometricOpWithArgs:
    """Unary isometric operation involving an n-dimensional seed array with a scalar or 1-dimensional vector,
    based on Bioconductor's ``DelayedArray::DelayedUnaryIsoOpWithArgs`` class.
    Only one n-dimensional array is involved here, hence the "unary" in the name.
    (Hey, I don't make the rules.)

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. We suggest supplying a floating-point ``value`` to avoid unexpected results from
    integer truncation or overflow.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end-users should not be interacting with ``UnaryIsometricOpWithArgs`` objects directly.

    Attributes:
        seed:
            Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        value (Union[float, ndarray]):
            A scalar or NumPy array with which to perform an operation on the ``seed``.

            If scalar, the operation is applied element-wise to all entries of ``seed``.

            If a 1-dimensional NumPy array, the operation is broadcast along the last dimension of ``seed``.

            If an n-dimensional NumPy array, the number of dimensions should be equal to the dmensionality of ``seed``.
            All dimensions should be of extent 1, except for exactly one dimension that should have extent equal to the
            corresponding dimension of ``seed``. The operation is then broadcast along that dimension.

        operation (str):
            String specifying the operation.

        right (bool, optional):
            Whether ``value`` is to the right of ``seed`` in the operation.
            If False, ``value`` is put to the left of ``seed``.
            Ignored for commutative operations in ``op``.
    """

    def __init__(
        self, seed, value: Union[float, ndarray], operation: OP, right: bool = True
    ):
        dummy = numpy.zeros(0, dtype=seed.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            if isinstance(value, ndarray):
                dummy = _execute(dummy, value[:0], operation)
            else:
                dummy = _execute(dummy, value, operation)
        dtype = dummy.dtype

        along = None
        if isinstance(value, ndarray):
            ndim = len(seed.shape)

            if len(value.shape) == 1:
                along = ndim - 1
            else:
                if len(value.shape) != ndim:
                    raise ValueError(
                        "length of 'value.shape' and 'seed.shape' should be equal"
                    )

                for i in range(ndim):
                    if value.shape[i] != 1:
                        if along is not None:
                            raise ValueError(
                                "no more than one entry of 'value.shape' should be greater than 1"
                            )
                        if seed.shape[i] != value.shape[i]:
                            raise ValueError(
                                "any entry of 'value.shape' that is not 1 should be equal to the corresponding entry of 'seed.shape'"  # noqa: E501
                            )
                        along = i

                if along is None:
                    value = value[(*([0] * ndim), ...)]

        self._seed = seed
        self._value = value
        self._op = operation
        self._right = right
        self._along = along
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``UnaryIsometricOpWithArgs`` object. As the name of the class suggests, this is the same as the
        ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the
            ``UnaryIsometricOpWithArgs`` object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        """Type of the ``UnaryIsometricOpWithArgs`` object. This may or may not be the same as the ``seed`` array,
        depending on how NumPy does the casting for the requested operation.

        Returns:
            dtype: NumPy type for the ``UnaryIsometricOpWithArgs`` contents.
        """
        return self._dtype

    @property
    def seed(self):
        """Get the underlying object satisfying the seed contract.

        Returns:
            The seed object.
        """
        return self._seed

    @property
    def operation(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Name of the operation.
        """
        return self._op

    @property
    def value(self) -> Union[float, ndarray]:
        """Get the other operand used in the operation.

        Returns:
            Union[float, ndarray]: The other operand.
            This can be a numeric scalar or a NumPy array.
        """
        return self._value

    @property
    def right(self) -> bool:
        """Is the :py:attr:`~value` applied to the right of the seed?

        Returns:
            bool: Whether to apply the operation to the right of the seed.
        """
        return self._right

    @property
    def along(self) -> Union[int, None]:
        """If :py:attr:`~value` is an array, this specifies the dimension of :py:attr:``~seed`` along which the array
        values are broadcast.

        Returns:
            Union[int, None]: Broadcasting dimension, or None if ``value`` is a scalar.
        """
        return self._along

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        target = create_dask_array(self._seed)
        operand = self._value
        if self._right:
            return _execute(target, operand, self._op)
        else:
            return _execute(operand, target, self._op)

    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        target = extract_array(self._seed, subset)

        subvalue = self._value
        if isinstance(subvalue, ndarray):
            if len(subvalue.shape) == 1:
                subvalue = subvalue[subset[-1]]
            else:
                resub = [slice(None)] * len(subset)
                subdim = self.along
                resub[subdim] = subset[subdim]
                subvalue = subvalue[(*resub,)]

        def f(s):
            if self._right:
                return _execute(s, subvalue, self._op)
            else:
                return _execute(subvalue, s, self._op)

        return _retry_single(target, f, target.shape)

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        return chunk_shape(self._seed)
