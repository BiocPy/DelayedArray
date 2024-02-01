from functools import singledispatch
from typing import Any, Tuple, Sequence
from numpy import ndarray
from biocutils.package_utils import is_package_installed

from .SparseNdarray import SparseNdarray
from .RegularTicks import RegularTicks
from .Grid import SimpleGrid, AbstractGrid

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def chunk_shape_to_grid(chunks: Sequence[int], shape: Tuple[int, ...], cost_factor: int) -> SimpleGrid:
    """
    Convert a chunk shape to a :py:class:`~delayedarray.Grid.SimpleGrid`.
    This assumes that the underlying array is split up into regular intervals
    on each dimension; the first chunk should start from zero, and only the
    last chunk may be of a different size (bounded by the dimension extent).

    Args:
        chunks:
            Chunk size for each dimension. These should be positive.

        shape:
            Extent of each dimension of the array. These should be non-negative
            and of the same length as ``chunks``.

        cost_factor:
            Cost factor for iterating over each element of the associated
            array. This is used to decide between iteration schemes and can be
            increased for more expensive types, e.g., file-backed arrays. As a
            reference, in-memory NumPy arrays are assigned a cost factor of 1.

    Returns:
        A ``SimpleGrid`` object with the chunk shape as the boundaries.
    """
    out = []
    for i, ch in enumerate(chunks):
        sh = shape[i]
        if sh == 0:
            out.append([])
        elif ch == sh:
            out.append([sh])
        else:
            out.append(RegularTicks(ch, sh))
    return SimpleGrid((*out,), cost_factor=cost_factor)


@singledispatch
def chunk_grid(x: Any) -> AbstractGrid:
    """
    Create a grid over the array, used to determine how a caller should iterate
    over that array. The intervals of the grid usually reflects a particular
    layout of the data on disk or in memory.

    Args:
        x: An array-like object.
    
    Returns:
        An instance of a :py:class:`~delayedarray.Grid.AbstractGrid`.
    """
    raise NotImplementedError("'chunk_grid(" + str(type(x)) + ")' has not yet been implemented")


@chunk_grid.register
def chunk_grid_ndarray(x: ndarray) -> SimpleGrid:
    """
    See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

    The cost factor for iteration is set to 1, which is considered the lowest
    cost for data extraction given that everything is stored in memory.
    """
    raw = [1] * len(x.shape)
    if x.flags.f_contiguous:
        raw[0] = x.shape[0]
    else:
        # Not sure how to deal with strided views here; not even sure how
        # to figure that out from NumPy flags. Guess we should just assume
        # that it's C-contiguous, given that most things are.
        raw[-1] = x.shape[-1]
    return chunk_shape_to_grid(raw, x.shape, cost_factor=1)


@chunk_grid.register
def chunk_grid_SparseNdarray(x: SparseNdarray) -> SimpleGrid:
    """
    See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

    The cost factor for iteration is set to 1.5. This is slightly higher than
    that of dense NumPy arrays as the ``SparseNdarray`` is a bit more expensive
    for random access on the first dimension.
    """
    raw = [1] * len(x.shape)
    raw[0] = x.shape[0]
    return chunk_shape_to_grid(raw, x.shape, cost_factor=1.5)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.

if is_package_installed("scipy"):
    import scipy.sparse as sp


    @chunk_grid.register
    def chunk_grid_csc_matrix(x: sp.csc_matrix) -> SimpleGrid:
        """
        See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

        The cost factor for iteration is set to 1.5. This is slightly higher
        than that of dense NumPy arrays as CSC matrices are a bit more
        expensive for random row access.
        """
        return chunk_shape_to_grid((x.shape[0], 1), x.shape, cost_factor=1.5)


    @chunk_grid.register
    def chunk_grid_csr_matrix(x: sp.csr_matrix) -> SimpleGrid:
        """
        See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

        The cost factor for iteration is set to 1.5. This is slightly higher
        than that of dense NumPy arrays as CSR matrices are a bit more
        expensive for random column access.
        """
        return chunk_shape_to_grid((1, x.shape[1]), x.shape, cost_factor=1.5)


    @chunk_grid.register
    def chunk_grid_coo_matrix(x: sp.coo_matrix) -> SimpleGrid:
        """
        See :py:meth:`~delayedarray.chunk_grid.chunk_grid`.

        The cost factor for iteration is set to 5, as any extraction from a COO
        matrix requires a full scan through all elements.
        """
        # ???? let's just do our best here, there's no nice way to access COO.
        return chunk_shape_to_grid(x.shape, x.shape, cost_factor=5)
