__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

from typing import Tuple, Union, Sequence


def sanitize_single_index(idx, shape: int) -> Sequence[int]:
    """Sanitize a single index sequence. This is called by :py:meth:`~delayedarray.utils.sanitize_indices`.

    Args:
        idx (Union[slice, Sequence]):
            Vector of indices to extract from a dimension.
            This can be a sequence of integers or a :py:class:`~slice`.

        shape (Tuple[int, ...]):
            Extent of the dimension to extract from.

    Returns:
        Sequence[int]: Integer sequences specifying the indices to extract.
        Any :py:class:`~slice` is converted into the appropriate :py:class:`~range` objects.
    """
    if isinstance(idx, slice):
        return range(*idx.indices(shape))
    return idx


def sanitize_indices(
    idx: Tuple[Union[slice, Sequence], ...], shape: Tuple[int, ...]
) -> Tuple[Sequence[int], ...]:
    """Sanitize indices for use in methods like :py:meth:`~delayedarray.interface.extract_dense_array`.

    Args:
        idx (Tuple[Union[slice, Sequence], ...]):
            Tuple where each entry defines a vector of indices to extract from a dimension.
            This can be a sequence of integers or a :py:class:`~slice`.

        shape (Tuple[int, ...]):
            Tuple containing the shape of the array to be extracted from.
            This should be the same length as ``idx``.

    Returns:
        Tuple[Sequence[int], ...]: Tuple of integer sequences specifying the indices to extract.
        Slices are converted into the appropriate :py:class:`~range` objects.
    """
    idx2 = []
    for i in range(len(idx)):
        idx2.append(sanitize_single_index(idx[i], shape[i]))
    return idx2
