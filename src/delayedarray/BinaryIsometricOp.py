import warnings
from typing import Tuple, Sequence, TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    import dask.array

from .UnaryIsometricOpWithArgs import OP, _execute
from .utils import create_dask_array, extract_array, _densify, chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class BinaryIsometricOp:
    """Binary isometric operation involving two n-dimensional seed arrays with the same dimension extents.
    This is based on Bioconductor's ``DelayedArray::DelayedNaryIsoOp`` class.

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. It is probably safest to cast at least one array to floating-point
    to avoid problems due to integer overflow.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end users should not be interacting with ``BinaryIsometricOp`` objects directly.

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

        ldummy = numpy.zeros(0, dtype=left.dtype)
        rdummy = numpy.zeros(0, dtype=right.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            dummy = _execute(ldummy, rdummy, operation)
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
            Tuple[int, ...]: Tuple of integers specifying the extent of each dimension of the ``BinaryIsometricOp``
            object.
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

    def __DelayedArray_dask__(self) -> "dask.array.core.Array":
        """See :py:meth:`~delayedarray.utils.create_dask_array`."""
        ls = create_dask_array(self._left)
        rs = create_dask_array(self._right)
        return _execute(ls, rs, self._op)

    def __DelayedArray_extract__(self, subset: Tuple[Sequence[int]]):
        """See :py:meth:`~delayedarray.utils.extract_array`."""
        ls = extract_array(self._left, subset)
        rs = extract_array(self._right, subset)

        try:
            output = _execute(ls, rs, self._op)
            if output.shape != self.shape:
                raise ValueError(
                    "operation on "
                    + str(type(seed))
                    + " does not return the expected shape"
                )
        except Exception as e:
            warnings.warn(str(e))
            ls = _densify(ls)
            rs = _densify(rs)
            output = _execute(ls, rs, self._op)

        return output

    def __DelayedArray_chunk__(self) -> Tuple[int]:
        """See :py:meth:`~delayedarray.utils.chunk_shape`."""
        lchunk = chunk_shape(self._left)
        rchunk = chunk_shape(self._right)

        # Not bothering with taking the lowest common denominator, as that
        # might be too aggressive and expanding to the entire matrix size.
        # We instead use the maximum chunk size (which might also expand, e.g.,
        # if you're combining column-major and row-major matrices; oh well).
        # Just accept that we'll probably need to break chunks during iteration.
        output = []
        for i in range(len(lchunk)):
            output.append(max(lchunk[i], rchunk[i]))

        return (*output,) 
