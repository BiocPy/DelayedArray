from functools import singledispatch

from .DelayedArray import DelayedArray


@singledispatch
def as_delayed_array(x: Any) -> DelayedArray:
    """Wrap an object in a :py:class:`~delayedarray.DelayedArray.DelayedArray`.
    Developers can implement methods for this generic to create DelayedArray
    subclasses that are specialized for their own seed objects.

    Args:
        x: 
            Any object satisfiying the seed contract, see documentation for
            :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

    Returns:
        A DelayedArray.
    """
    return DelayedArray(x)


@as_delayed_array.register
def as_delayed_array(x: DelayedArray):
    """See :py:meth:`~delayedarray.as_delayed_array.as_delayed_array`."""
    return x
