from functools import singledispatch
from typing import Any

from .DelayedArray import DelayedArray


@singledispatch
def wrap(x: Any) -> DelayedArray:
    """Create a :py:class:`~delayedarray.DelayedArray.DelayedArray` from an
    object satisfying the seed contract.  Developers can implement methods for
    this generic to create ``DelayedArray`` subclasses based on the seed type.

    Args:
        x: 
            Any object satisfiying the seed contract, see documentation for
            :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

    Returns:
        A ``DelayedArray`` or one of its subclasses.
    """
    return DelayedArray(x)


@wrap.register
def wrap_DelayedArray(x: DelayedArray):
    """See :py:meth:`~delayedarray.wrap.wrap`."""
    return x
