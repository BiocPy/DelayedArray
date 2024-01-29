import warnings
from typing import Callable, Tuple, Union, Sequence
import numpy
from numpy import ndarray

from .DelayedOp import DelayedOp
from .SparseNdarray import SparseNdarray
from ._isometric import ISOMETRIC_OP_WITH_ARGS, _execute, _infer_along_with_args
from ._subset import _spawn_indices
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .create_dask_array import create_dask_array
from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .is_masked import is_masked

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


class UnaryIsometricOpWithArgs(DelayedOp):
    """Unary isometric operation involving an n-dimensional seed array with a scalar or 1-dimensional vector,
    based on Bioconductor's ``DelayedArray::DelayedUnaryIsoOpWithArgs`` class.
    Only one n-dimensional array is involved here, hence the "unary" in the name.
    (Hey, I don't make the rules.)

    The data type of the result is determined by NumPy casting given the ``seed`` and ``value``
    data types. We suggest supplying a floating-point ``value`` to avoid unexpected results from
    integer truncation or overflow.

    This class is intended for developers to construct new :py:class:`~delayedarray.DelayedArray.DelayedArray`
    instances. In general, end-users should not be interacting with ``UnaryIsometricOpWithArgs`` objects directly.
    """

    def __init__(self, seed, value: Union[float, ndarray], operation: ISOMETRIC_OP_WITH_ARGS, right: bool = True):
        """
        Args:
            seed:
                Any object satisfying the seed contract, see
                :py:meth:`~delayedarray.DelayedArray.DelayedArray` for details.

            value:
                A scalar or NumPy array with which to perform an operation on the ``seed``.

                If scalar, the operation is applied element-wise to all entries of ``seed``.

                If a 1-dimensional NumPy array, the operation is broadcast along the last dimension of ``seed``.

                If an n-dimensional NumPy array, the number of dimensions should be equal to the dmensionality of ``seed``.
                All dimensions should be of extent 1, except for exactly one dimension that should have extent equal to the
                corresponding dimension of ``seed``. The operation is then broadcast along that dimension.

            operation:
                String specifying the operation.

            right:
                Whether ``value`` is to the right of ``seed`` in the operation.
                If False, ``value`` is put to the left of ``seed``. Ignored
                for commutative operations in ``op``.
        """

        along = _infer_along_with_args(seed.shape, value)
        if along is None and isinstance(value, ndarray):
            ndim = len(value.shape)
            value = value[(*([0] * ndim),)]

        with warnings.catch_warnings():  # silence warnings from divide by zero.
            warnings.simplefilter("ignore")
            if isinstance(value, ndarray):
                dummy = numpy.zeros(value.shape, dtype=seed.dtype)
                dummy = _execute(dummy, value, operation)
            else:
                dummy = numpy.zeros(1, dtype=seed.dtype)
                dummy = _execute(dummy, value, operation)

        self._seed = seed
        self._value = value
        self._op = operation
        self._right = right
        self._along = along
        self._dtype = dummy.dtype
        self._sparse = is_sparse(self._seed) and (dummy == 0).all()

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple of integers specifying the extent of each dimension of this
            object. This should be the same as ``seed``.
        """
        return self._seed.shape

    @property
    def dtype(self) -> numpy.dtype:
        """
        Returns:
            NumPy type for the data after the operation was applied. This may
            or may not be the same as the ``seed`` array, depending on how
            NumPy does the casting for the requested operation.
        """
        return self._dtype

    @property
    def seed(self):
        """
        Returns:
            The seed object.
        """
        return self._seed

    @property
    def operation(self) -> ISOMETRIC_OP_WITH_ARGS:
        """
        Returns:
            Name of the operation.
        """
        return self._op

    @property
    def value(self) -> Union[float, ndarray]:
        """
        Returns:
            The other operand used in the operation.
        """
        return self._value

    @property
    def right(self) -> bool:
        """
        Returns:
            Whether the operation was applied to the right of the seed.
        """
        return self._right

    @property
    def along(self) -> Union[int, None]:
        """
        Returns:
            The dimension of :py:attr:``~seed`` along which the array values
            are broadcast, for an array :py:attr:`~value`. Otherwise None, if
            ``value`` is a scalar.
        """
        return self._along


def _extract_array(x: UnaryIsometricOpWithArgs, subset: Tuple[Sequence[int], ...], f: Callable): 
    target = f(x._seed, subset)

    subvalue = x._value
    if isinstance(subvalue, ndarray) and not subvalue is numpy.ma.masked:
        if subset is None:
            subset = _spawn_indices(x.shape)
        if len(subvalue.shape) == 1:
            subvalue = subvalue[subset[-1]]
        else:
            resub = [slice(None)] * len(subset)
            subdim = x.along
            resub[subdim] = subset[subdim]
            subvalue = subvalue[(*resub,)]

    if x._right:
        return _execute(target, subvalue, x._op)
    else:
        return _execute(subvalue, target, x._op)


@extract_dense_array.register
def extract_dense_array_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs, subset: Tuple[Sequence[int], ...]) -> numpy.ndarray:
    """See :py:meth:`~delayedarray.extract_dense_array.extract_dense_array`."""
    return _extract_array(x, subset, extract_dense_array)


@extract_sparse_array.register
def extract_sparse_array_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs, subset: Tuple[Sequence[int], ...]) -> SparseNdarray:
    """See :py:meth:`~delayedarray.extract_sparse_array.extract_sparse_array`."""
    return _extract_array(x, subset, extract_sparse_array)


@create_dask_array.register
def create_dask_array_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs):
    """See :py:meth:`~delayedarray.create_dask_array.create_dask_array`."""
    target = create_dask_array(x._seed)
    operand = x._value
    if x._right:
        return _execute(target, operand, x._op)
    else:
        return _execute(operand, target, x._op)


@chunk_shape.register
def chunk_shape_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs):
    """See :py:meth:`~delayedarray.chunk_shape.chunk_shape`."""
    return chunk_shape(x._seed)


@is_sparse.register
def is_sparse_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs):
    """See :py:meth:`~delayedarray.is_sparse.is_sparse`."""
    return x._sparse


@is_masked.register
def is_masked_UnaryIsometricOpWithArgs(x: UnaryIsometricOpWithArgs):
    """See :py:meth:`~delayedarray.is_masked.is_masked`."""
    return is_masked(x._seed) or numpy.ma.isMaskedArray(x._value) or x._value is numpy.ma.masked

