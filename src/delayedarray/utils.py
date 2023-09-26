from typing import Optional, Tuple, Sequence, TYPE_CHECKING
from numpy import array, ndarray, ix_
from scipy.sparse import sparray, coo_matrix, csr_matrix, csc_matrix
import warnings

if TYPE_CHECKING:
    import dask.array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def create_dask_array(seed) -> "dask.array.core.Array":
    """Create a dask array containing the delayed operations. This requires
    the dask package to be installed and will load it into the current session. 

    Args:
        seed: Any object that can be converted into a dask array, or has a
            ``__DelayedArray_dask__`` method that returns a dask array, or
            is already a dask array.

    Returns:
        Array: dask array, possibly containing delayed operations.
    """

    if hasattr(seed, "__DelayedArray_dask__"):
        return seed.__DelayedArray_dask__()

    import dask.array

    if isinstance(seed, dask.array.core.Array):
        return seed
    else:
        return dask.array.from_array(seed)


def chunk_shape(seed) -> Tuple[int]:
    """Get the dimensions of the array chunks. These define the preferred
    blocks with which to iterate over the array in each dimension.

    Args:
        seed: Any seed object.
    
    Returns:
        Tuple of integers containing the shape of the chunk.
    """
    if hasattr(seed, "__DelayedArray_chunk__"):
        return seed.__DelayedArray_chunk__()

    if isinstance(seed, ndarray):
        sh = list(seed.shape)
        if seed.flags.f_contiguous:
            for i in range(1, len(sh)):
                sh[i] = 1
        else:
            # Not sure how to deal with strided views here; not even sure how
            # to figure that out from NumPy flags. Guess we should just assume
            # that it's C-contiguous, given that most things are.
            for i in range(len(sh) - 1):
                sh[i] = 1
        return (*sh,)

    if isinstance(seed, csc_matrix):
        return (seed.shape[0], 1)
    elif isinstance(seed, csr_matrix):
        return (1, seed.shape[1])

    # Guess we should return something.
    return seed.shape


def guess_iteration_block_size(seed, dimension: int, memory: int = 10000000) -> int:
    """Guess the best block size for iterating over the matrix on a certain
    dimension.  This assumes that, in each iteration, an entire block of
    observations is extracted involving the full extent of all dimensions other
    than the one being iterated over. This block is used for processing before
    extracting the next block of elements.

    Args:
        seed: Any seed object.

        dimension: Dimension to iterate over.

        memory: Available memory in bytes, to hold a single block in memory.

    Returns:
        Size of the block on the iteration dimension.
    """
    num_elements = memory / seed.dtype.itemsize
    shape = seed.shape

    prod_other = 1
    for i, s in enumerate(shape):
        if i != dimension:
            prod_other *= s 

    ideal = int(num_elements / prod_other)
    if ideal == 0:
        return 1

    curdim = chunk_shape(seed)[dimension]
    if ideal <= curdim:
        return ideal

    return int(ideal / curdim) * curdim


def is_sparse(seed) -> bool:
    """Determine whether a seed object contains sparse data. This calls
    :py:meth:`~delayedarray.DelayedArray.DelayedArray.__DelayedArray_sparse__`
    if available, otherwise it falls back to some hard-coded rules.

    Args:
        seed: Any seed object.

    Returns:
        Whether the seed contains sparse data.
    """
    if hasattr(seed, "__DelayedArray_sparse__"):
        return seed.__DelayedArray_sparse__()

    if isinstance(seed, sparray) or isinstance(seed, csc_matrix) or isinstance(seed, csr_matrix) or isinstance(seed, coo_matrix):
        return True

    return False
