import warnings
from typing import Callable, Optional, Tuple, Sequence, TYPE_CHECKING
import numpy
if TYPE_CHECKING:
    import dask.array

from .DelayedOp import DelayedOp
from ._isometric import ISOMETRIC_OP_WITH_ARGS, _execute
from .extract_dense_array import extract_dense_array, _sanitize_to_fortran
from .extract_sparse_array import extract_sparse_array
from .utils import create_dask_array, chunk_shape, is_sparse

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class BinaryIsometricOp(DelayedOp):
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

    def __init__(self, left, right, operation: ISOMETRIC_OP_WITH_ARGS):
        if left.shape != right.shape:
            raise ValueError("'left' and 'right' shapes should be the same")

        ldummy = numpy.zeros(1, dtype=left.dtype)
        rdummy = numpy.zeros(1, dtype=right.dtype)
        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            dummy = _execute(ldummy, rdummy, operation)
        dtype = dummy.dtype

        self._left = left
        self._right = right
        self._op = operation
        self._dtype = dtype
        self._sparse = is_sparse(self._left) and is_sparse(self._right) and dummy[0] == 0

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

    def __DelayedArray_sparse__(self) -> bool:
        """See :py:meth:`~delayedarray.utils.is_sparse`."""
        return self._sparse

 
def _extract_array(x: BinaryIsometricOp, subset: Optional[Tuple[Sequence[int]]], f: Callable):
    ls = f(x._left, subset)
    rs = f(x._right, subset)
    return _execute(ls, rs, x._op)


@extract_dense_array.register
def extract_dense_array_BinaryIsometricOp(x: BinaryIsometricOp, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    out = _extract_array(x, subset, extract_dense_array)
    return _sanitize_to_fortran(out)


@extract_sparse_array.register
def extract_sparse_array_BinaryIsometricOp(x: BinaryIsometricOp, subset: Optional[Tuple[Sequence[int]]] = None):
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)
