from typing import Callable, Optional
import math

from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

def guess_iteration_block_size(x, dimension, memory: int = 10000000) -> int:
    """
    Soft-deprecated alias for :py:func:`~choose_block_size_for_1d_iteration`.
    """
    return choose_block_size_for_1d_iteration(x, dimension, memory)

def choose_block_size_for_1d_iteration(x, dimension: int, memory: int = 10000000) -> int:
    """
    Choose a block size for iterating over an array on a certain dimension,
    see `~apply_over_dimension` for more details.

    Args:
        x: An array-like object.

        dimension: Dimension to iterate over.

        memory: Available memory in bytes, to hold a single block in memory.

    Returns:
        Size of the block on the iteration dimension.
    """
    num_elements = memory / x.dtype.itemsize
    shape = x.shape

    prod_other = 1
    for i, s in enumerate(shape):
        if i != dimension:
            prod_other *= s

    ideal = int(num_elements / prod_other)
    if ideal == 0:
        return 1

    curdim = chunk_shape(x)[dimension]
    if ideal <= curdim:
        return ideal

    return int(ideal / curdim) * curdim


def apply_over_dimension(x, dimension: int, fun: Callable, block_size: Optional[int] = None, allow_sparse: bool = False) -> list:
    """
    Iterate over an array on a certain dimension. At each iteration, the block
    of observations consists of the full extent of all dimensions other than
    the one being iterated over. We apply a user-provided function and collect
    the results before proceeding to the next block.

    Args:
        x: An array-like object.

        dimension: Dimension to iterate over.

        fun:
            Function to apply to each block. This should accept two arguments;
            the first is a tuple containing the start/end of the current block
            on the chosen ``dimension``, and the second is the block contents.
            Each block is typically provided as a :py:class:`~numpy.ndarray`.

        block_size:
            Size of the block on the iteration dimension. If None, this is
            chosen by :py:func:`~choose_block_size_for_1d_iteration`.

        allow_sparse:
            Whether to allow extraction of sparse subarrays. If true and
            ``x`` contains a sparse array, the block contents are instead
            represented by a :py:class:`~SparseNdarray.SparseNdarray`.

    Returns:
        List containing the output of ``fun`` on each block.
    """
    if block_size is None:
        block_size = choose_block_size_for_1d_iteration(x, dimension)

    limit = x.shape[dimension]
    tasks = math.ceil(limit / block_size)
    components = [range(n) for n in x.shape]
    if allow_sparse and is_sparse(x):
        extractor = extract_sparse_array
    else:
        extractor = extract_dense_array

    collected = []
    for job in range(tasks):
        start = job * block_size
        end = min(start + block_size, limit)
        components[dimension] = range(start, end)
        subset = (*components,)
        output = fun((start, end), extractor(x, subset))
        collected.append(output)

    return collected
