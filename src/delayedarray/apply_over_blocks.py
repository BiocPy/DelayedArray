from typing import Callable, Optional, Tuple
import math

from .chunk_grid import chunk_grid
from .Grid import AbstractGrid
from .is_sparse import is_sparse
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def apply_over_blocks(x, fun: Callable, allow_sparse: bool = False, grid: Optional[AbstractGrid] = None, buffer_size: int = 1e8) -> list:
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

        allow_sparse:
            Whether to allow extraction of sparse subarrays. If true and ``x``
            contains a sparse array, the block contents are instead represented
            by a :py:class:`~delayedarray.SparseNdarray.SparseNdarray`.

        grid:
            Grid to subdivide ``x`` for iteration. Specifically, iteration will
            attempt to extract blocks that are aligned with the grid boundaries,
            e.g., to optimize extraction of chunked data. Defaults to the output
            of :py:func:`~delayedarray.chunk_grid.chunk_grid` on ``x``.

        buffer_size: 
            Buffer_size in bytes, to hold a single block per iteration. Larger
            values generally improve speed at the cost of memory.

    Returns:
        List containing the output of ``fun`` on each block.
    """
    if grid is None:
        grid = chunk_grid(x)

    if allow_sparse and is_sparse(x):
        extractor = extract_sparse_array
    else:
        extractor = extract_dense_array

    dims = (*range(len(x.shape)),)
    collected = []
    buffer_elements = buffer_size // x.dtype.itemsize

    for job in grid.iterate(dims, buffer_elements = buffer_elements):
        subsets = (*(range(s, e) for s, e in job),)
        output = fun(job, extractor(x, subsets))
        collected.append(output)

    return collected
