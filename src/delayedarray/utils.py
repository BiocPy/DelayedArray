from typing import Optional, Tuple, Sequence
from dask.array import from_array, Array
from numpy import array, ndarray, ix_
from scipy.sparse import coo_matrix
import warnings

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def create_dask_array(seed) -> Array:
    """Create a dask array containing the delayed operations.

    Args:
        seed: Any object that can be converted into a dask array, or has a
            ``__DelayedArray_dask__`` method that returns a dask array.

    Returns:
        Array: dask array, possibly containing delayed operations.
    """

    if hasattr(seed, "__DelayedArray_dask__"):
        return seed.__DelayedArray_dask__()
    elif isinstance(seed, Array):
        return seed
    else:
        return from_array(seed)


def _spawn_indices(shape):
    raw = []
    for s in shape:
        raw.append(range(s))
    return (*raw,)


def extract_array(seed, subset: Optional[Tuple[Sequence[int]]] = None):
    """Extract the realized contents (or a subset thereof) into some NumPy-like array. For delayed operations, this
    attempts to preserve the class of the seed, which may improve performance when dealing with objects like sparse
    matrices; if the class does not support the operation, this function will fall back to creating a NumPy array and
    applying the operation on that instead.

    Args:
        seed: Any object that supports slicing by :py:meth:`~numpy.ix_`, or has a
            ``__DelayedArray_extract__`` method that accepts ``subset`` and
            returns an array-like object containing the outer product of the subsets.

        subset (Tuple[Sequence[int]], optional): Tuple of length equal to the number of dimensions,
            each containing a sorted and unique sequence of integers specifying the
            elements of each dimension to extract. If None, all elements are extracted
            from all dimensions.

    Returns:
        Some array-like object where all delayed operations are evaluated
        for the specified ``subset``.
    """

    if hasattr(seed, "__DelayedArray_extract__"):
        if subset is None:
            subset = _spawn_indices(seed.shape)
        output = seed.__DelayedArray_extract__(subset)
    else:
        noop = True
        if subset is not None:
            for i, s in enumerate(seed.shape):
                cursub = subset[i]
                if len(cursub) != s:
                    noop = False
                    break

                for j in range(s):
                    if cursub[j] != j:
                        noop = False
                        break
                if not noop:
                    break
        else:
            subset = _spawn_indices(seed.shape)

        if noop:
            output = seed
        else:
            output = seed[ix_(*subset)]

    outshape = output.shape
    for i, s in enumerate(subset):
        if len(s) != outshape[i]:
            raise ValueError(
                "extract_array on "
                + str(type(seed))
                + " does not return the expected shape"
            )

    # CSC/CSR matrices get coerced to COO matrices during arithmetic.
    # Unfortunately, COO matrices don't support subscripting. Hence,
    # we need to always coerce it back to a CSC matrix, just in case
    # the caller needs to do some subscripting.
    if isinstance(output, coo_matrix):
        output = output.tocsc()

    return output


def _densify(seed):
    if isinstance(seed, ndarray):
        return seed

    if hasattr(seed, "toarray"):
        output = seed.toarray()
    elif hasattr(seed, "__array__"):
        output = array(seed)
    else:
        raise ValueError(
            "don't know how to convert " + str(type(seed)) + " to a NumPy array"
        )

    if seed.shape != output.shape:
        raise ValueError(
            "conversion to NumPy array for "
            + str(type(seed))
            + " does not return the expected shape"
        )
    return output


def _retry_single(seed, f, expected_shape):
    try:
        output = f(seed)
        if output.shape != expected_shape:
            raise ValueError(
                "operation on "
                + str(type(seed))
                + " does not return the expected shape"
            )
    except Exception as e:
        warnings.warn(str(e))
        output = f(_densify(seed))
    return output
