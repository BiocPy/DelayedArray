import operator
import warnings
from typing import Literal, Tuple, Union

import numpy
from numpy import ndarray
from dask.array.core import Array

from .utils import _create_dask_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

OP = Literal["+", "-", "/", "*", "//", "%", "**"]


def _choose_operator(op: OP, inplace: bool = False):
    if op == "+":
        if inplace:
            return operator.iadd
        else:
            return operator.add
    elif op == "-":
        if inplace:
            return operator.isub
        else:
            return operator.sub
    elif op == "*":
        if inplace:
            return operator.imul
        else:
            return operator.mul
    elif op == "/":
        if inplace:
            return operator.itruediv
        else:
            return operator.truediv
    elif op == "//":
        if inplace:
            return operator.ifloordiv
        else:
            return operator.floordiv
    elif op == "%":
        if inplace:
            return operator.imod
        else:
            return operator.mod
    elif op == "**":
        if inplace:
            return operator.ipow
        else:
            return operator.pow
    else:
        raise ValueError("unknown operation '" + op + "'")


class UnaryIsometricOpWithArgs:
    """Unary isometric operation involving an n-dimensional seed array with a scalar or 1-dimensional vector.
    This is based on Bioconductor's ``DelayedArray::DelayedUnaryIsoOpWithArgs`` class.
    Only one n-dimensional array is involved here, hence the "unary" in the name.
    (I don't make the rules.)

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. We suggest supplying a floating-point ``value`` to avoid unexpected results from
    integer truncation or overflow.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    In general, end-users should not be interacting with ``UnaryIsometricOpWithArgs`` objects directly.

    Attributes:
        seed:
            Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        value (Union[float, ndarray]):
            A scalar or 1-dimensional array with which to perform an operation on the ``seed``.

        op (str):
            String specifying the operation.

        right (bool, optional):
            Whether ``value`` is to the right of ``seed`` in the operation.
            If False, ``value`` is put to the left of ``seed``.
            Ignored for commutative operations in ``op``.

        along (int, optional):
            Dimension along which the ``value`` is to be added, if ``value`` is a
            1-dimensional array. This assumes that ``value`` is of length equal to the dimension's
            extent. Ignored if ``value`` is a scalar.
    """

    def __init__(
        self,
        seed,
        value: Union[float, ndarray],
        op: OP,
        right: bool = True,
        along: int = 0,
    ):
        f = _choose_operator(op)

        dummy = numpy.zeros(0, dtype=seed.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            if isinstance(value, ndarray):
                dummy = f(dummy, value[:0])
            else:
                dummy = f(dummy, value)
        dtype = dummy.dtype

        if isinstance(value, ndarray):
            if along < 0 or along >= len(seed.shape):
                raise ValueError(
                    "'along' should be non-negative and less than the dimensionality of 'seed'"
                )
            if len(value) != seed.shape[along]:
                raise ValueError(
                    "length of array 'value' should equal the 'along' dimension extent in 'seed'"
                )

        self._seed = seed
        self._value = value
        self._op = op
        self._right = right
        self._along = along
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``UnaryIsometricOpWithArgs`` object. As the name of the class suggests, this is the same as the
        ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``UnaryIsometricOpWithArgs`` object.
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

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed operation.

        Returns:
            Array: dask array with the delayed subset.
        """
        target = _create_dask_array(self._seed)

        operand = self._value
        if (
            isinstance(self._value, numpy.ndarray)
            and self._along != len(self.shape) - 1
        ):
            dims = [1, 1, 1]
            dims[self._along] = self.shape[self._along]
            operand = operand.reshape(*dims)

        f = _choose_operator(self._op)
        if self._right:
            return f(target, operand)
        else:
            return f(operand, target)
