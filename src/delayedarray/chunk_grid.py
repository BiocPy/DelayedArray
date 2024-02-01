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


def _chunk_shape_to_grid(chunks: Sequence[int], shape: Tuple[int, ...], cost_factor: int):
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
def chunk_grid_ndarray(x: ndarray):
    """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`."""
    raw = [1] * len(x.shape)
    if x.flags.f_contiguous:
        raw[0] = x.shape[0]
    else:
        # Not sure how to deal with strided views here; not even sure how
        # to figure that out from NumPy flags. Guess we should just assume
        # that it's C-contiguous, given that most things are.
        raw[-1] = x.shape[-1]
    return _chunk_shape_to_grid(raw, x.shape, cost_factor=1)


@chunk_grid.register
def chunk_grid_SparseNdarray(x: SparseNdarray):
    """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`."""
    raw = [1] * len(x.shape)
    raw[0] = x.shape[0]
    return _chunk_shape_to_grid(raw, x.shape, cost_factor=1.5)


# If scipy is installed, we add all the methods for the various scipy.sparse matrices.

if is_package_installed("scipy"):
    import scipy.sparse as sp


    @chunk_grid.register
    def chunk_grid_csc_matrix(x: sp.csc_matrix):
        """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`."""
        return _chunk_shape_to_grid((x.shape[0], 1), x.shape, cost_factor=1.5)


    @chunk_grid.register
    def chunk_grid_csr_matrix(x: sp.csr_matrix):
        """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`."""
        return _chunk_shape_to_grid((1, x.shape[1]), x.shape, cost_factor=1.5)


    @chunk_grid.register
    def chunk_grid_coo_matrix(x: sp.coo_matrix):
        """See :py:meth:`~delayedarray.chunk_grid.chunk_grid`."""
        # ???? let's just do our best here, there's no nice way to access COO.
        return _chunk_shape_to_grid(x.shape, x.shape, cost_factor=1.5)
