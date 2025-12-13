from typing import Optional

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


old_buffer_size = 1e8


def default_buffer_size(buffer_size: Optional[int] = None) -> int:
    """
    Get or set the default buffer size used by :py:func:`~delayedarray.apply_over_blocks.apply_over_blocks`,
    :py:func:`~delayedarray.apply_over_dimension.apply_over_dimension`, etc.

    Args:
        buffer_size:
            Buffer size in bytes.
            The buffer is typically used to load a block of an array in memory for further processing.
            Alternatively ``None``.

    Returns:
        If ``buffer_size = None``, the current default buffer size is returned.

        If ``buffer_size`` is an integer, the default buffer size is set to this value, and the previous buffer size is returned.
    """
    global old_buffer_size
    if buffer_size is None:
        return old_buffer_size

    previous = old_buffer_size
    old_buffer_size = buffer_size
    return previous
