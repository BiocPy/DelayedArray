from typing import Callable, Optional, Tuple
import math

from .chunk_shape import chunk_shape
from .is_sparse import is_sparse
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def choose_block_shape_for_iteration(x, buffer_size: int = 10000000) -> Tuple[int, ...]:
    """
    Choose the block dimensions for blockwise iteration through an array, see
    `~apply_over_blocks` for details.

    Args:
        x: An array-like object.

        dimension: Dimension to iterate over.

        buffer_size: 
            Buffer_size in bytes, to hold a single block per iteration. Larger
            values generally improve speed at the cost of memory.

    Returns:
        Dimensions of the blocks. All values are guaranteed to be positive,
        even if the extent of any dimension of ``x`` is zero.
    """
    # Checking for empty dimensions and bailing out if we find any.
    for d in x.shape:
        if d == 0:
            return (*(max(1, d) for d in x.shape),)

    num_elements = buffer_size / x.dtype.itemsize
    chunk_dims = chunk_shape(x)
    block_size = 1
    for s in chunk_dims:
        block_size *= s

    block_dims = list(chunk_dims)

    if num_elements > block_size:
        # Going from the first dimension and increasing the block size.
        for i, y in enumerate(block_dims):
            multiple = int(num_elements / block_size)
            if multiple <= 1:
                break
            block_size /= y
            block_dims[i] = min(multiple * y, x.shape[i])
            block_size *= block_dims[i]

    elif num_elements < block_size:
        # Going from the last dimension and decreasing the block size.
        for i in range(len(block_dims) - 1, -1, -1):
            block_size_other = block_size / block_dims[i]
            multiple = int(num_elements / block_size_other)
            if multiple >= 1:
                block_dims[i] = multiple
                break
            block_size = block_size_other
            block_dims[i] = 1

    return (*block_dims,)


def apply_over_blocks(x, fun: Callable, block_shape: Optional[Tuple] = None, allow_sparse: bool = False, buffer_size: int = 1e8) -> list:
    """
    Iterate over an array by blocks. We apply a user-provided function and
    collect the results before proceeding to the next block.

    Args:
        x: An array-like object.

        fun:
            Function to apply to each block. This should accept two arguments;
            the first is a list containing the start/end of the current block
            on each dimension, and the second is the block contents. Each
            block is typically provided as a :py:class:`~numpy.ndarray`.

        block_shape:
            Dimensions of the block. All entries should be positive, even for
            zero-extent dimensions of ``x``. If None, this is chosen by
            :py:func:`~choose_block_shape_for_iteration`.

        allow_sparse:
            Whether to allow extraction of sparse subarrays. If true and
            ``x`` contains a sparse array, the block contents are instead
            represented by a :py:class:`~SparseNdarray.SparseNdarray`.

        buffer_size: 
            Buffer_size in bytes, to hold a single block per iteration. Larger
            values generally improve speed at the cost of memory. Only used
            if ``block_shape`` is not provided.

    Returns:
        List containing the output of ``fun`` on each block.
    """
    if block_shape is None:
        block_shape = choose_block_shape_for_iteration(x, buffer_size = buffer_size)

    num_tasks_total = 1
    num_tasks_by_dim = []
    for i, d in enumerate(x.shape):
        curtasks = math.ceil(d / block_shape[i])
        num_tasks_by_dim.append(curtasks)
        num_tasks_total *= curtasks

    if allow_sparse and is_sparse(x):
        extractor = extract_sparse_array
    else:
        extractor = extract_dense_array

    collected = []
    for job in range(num_tasks_total):
        subsets = []
        position = []
        counter = job
        for i, d in enumerate(num_tasks_by_dim):
            curtask = counter % d
            start = curtask * block_shape[i]
            end = min(start + block_shape[i], x.shape[i])
            position.append((start, end))
            subsets.append(range(start, end))
            counter //= d
        output = fun(position, extractor(x, (*subsets,)))
        collected.append(output)

    return collected
