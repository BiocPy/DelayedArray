from .DelayedArray import DelayedArray
from .DelayedOp import DelayedOp


def is_pristine(x) -> bool:
    """Determine whether an object is pristine, i.e., has no delayed operations.

    Args:
        x: Some array-like object.

    Returns:
        Whether ``x`` is a :py:class:`~delayedarray.DelayedArray.DelayedArray`
        containing delayed operations.
    """
    if isinstance(x, DelayedArray):
        x = x.seed
    return not isinstance(x, DelayedOp)
