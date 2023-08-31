from typing import Literal, Tuple

import numpy
from dask.array.core import Array
from numpy import dtype, zeros

from .utils import _create_dask_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

OP = Literal[
    "log",
    "log1p",
    "log2",
    "log10",
    "exp",
    "expm1",
    "sqrt",
    "abs",
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


def _choose_operator(op: OP):
    return getattr(numpy, op)


class UnaryIsometricOpSimple:
    """Delayed unary isometric operation involving an n-dimensional seed array with no additional arguments,
    similar to Bioconductor's ``DelayedArray::DelayedUnaryIsoOpStack`` class.
    This is used for simple mathematical operations like NumPy's :py:meth:`~numpy.log`.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. End-users should not be interacting with ``UnaryIsometricOpSimple`` objects directly.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        operation (str):
            String specifying the unary operation.
    """

    def __init__(self, seed, operation: OP):
        f = _choose_operator(operation)
        dummy = f(zeros(1, dtype=seed.dtype))

        self._seed = seed
        self._op = operation
        self._dtype = dummy.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the ``UnaryIsometricOpSimple`` object. As the name of the class suggests, this is the same as the
        ``seed`` array.

        Returns:
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``UnaryIsometricOpSimple``
            object.
        """
        return self._seed.shape

    @property
    def dtype(self) -> dtype:
        """Type of the ``UnaryIsometricOpSimple`` object. This may or may not be the same as the ``seed`` array,
        depending on how NumPy does the casting for the requested operation.

        Returns:
            dtype: NumPy type for the ``UnaryIsometricOpSimple`` contents.
        """
        return self._dtype

    def as_dask_array(self) -> Array:
        """Create a dask array containing the delayed unary operation.

        Returns:
            Array: dask array with the delayed unary operation.
        """
        target = _create_dask_array(self._seed)
        f = _choose_operator(self._op)
        return f(target)

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
