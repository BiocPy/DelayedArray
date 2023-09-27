from .chunk_shape import chunk_shape

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def guess_iteration_block_size(x, dimension: int, memory: int = 10000000) -> int:
    """Guess the best block size for iterating over an array on a certain
    dimension.  This assumes that, in each iteration, an entire block of
    observations is extracted involving the full extent of all dimensions other
    than the one being iterated over. This block is used for processing before
    extracting the next block of elements.

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
