import warnings
from typing import Tuple

import numpy
from dask.array.core import Array

from .UnaryIsometricOpWithArgs import OP, _choose_operator
from .utils import _create_dask_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class BinaryIsometricOp:
    """Binary isometric operation involving two n-dimensional seed arrays with the same dimension extents.
    This is based on Bioconductor's ``DelayedArray::DelayedNaryIsoOp`` class.

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. It is probably safest to cast at least one array to floating-point
    to avoid problems due to integer overflow.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray` instances.
    In general, end users should not be interacting with ``BinaryIsometricOp`` objects directly.

    Attributes:
        left:
            Any object satisfying the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        right:
            Any object of the same dimensions as ``left`` that satisfies the seed contract,
            see :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

        operation (str):
            String specifying the operation.
    """

    def __init__(self, left, right, operation: OP):
        if left.shape != right.shape:
            raise ValueError("'left' and 'right' shapes should be the same")

        f = _choose_operator(operation)
        ldummy = numpy.zeros(0, dtype=left.dtype)
        rdummy = numpy.zeros(0, dtype=right.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            dummy = f(ldummy, rdummy)
        dtype = dummy.dtype

        self._left = left
        self._right = right
        self._op = operation
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``BinaryIsometricOp`` object. As the name of the class suggests, this is the same as the
        ``left`` and ``right`` objects.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``BinaryIsometricOp`` object.
        """
        return self._left.shape

    @property
    def dtype(self) -> numpy.dtype:
        """Type of the ``BinaryIsometricOp`` object. This may or may not be the same as the ``left`` or ``right``
        objects, depending on how NumPy does the casting for the requested operation.

        Returns:
            dtype: NumPy type for the ``BinaryIsometricOp`` contents.
        """
        return self._dtype

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed operation.

        Returns:
            Array: dask array with the delayed subset.
        """
        f = _choose_operator(self._op)
        return f(_create_dask_array(self._left), _create_dask_array(self._right))

    @property
    def left(self):
        """Get the left operand satisfying the seed contract.

        Returns:
            The seed object on the left-hand-side of the operation.
        """
        return self._left

    @property
    def right(self):
        """Get the right operand satisfying the seed contract.

        Returns:
            The seed object on the right-hand-side of the operation.
        """
        return self._right

    @property
    def operation(self) -> str:
        """Get the name of the operation.

        Returns:
            str: Name of the operation.
        """
        return self._op
