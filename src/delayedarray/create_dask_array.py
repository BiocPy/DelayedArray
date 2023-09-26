from functools import singledispatch
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    import dask.array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


@singledispatch
def create_dask_array(x: Any) -> "dask.array.core.Array":
    """Create a dask array containing the delayed operations, assuming
    the **dask** package is installed.

    Args:
        x: Any array-like object.

    Returns:
        A dask array, possibly containing delayed operations.
    """
    import dask.array
    if isinstance(x, dask.array.core.Array):
        return x 
    else:
        return dask.array.from_array(x)
