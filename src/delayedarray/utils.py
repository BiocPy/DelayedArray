__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"

from dask.array import from_array


def _create_dask_array(seed):
    if hasattr(seed, "as_dask_array"):
        return seed.as_dask_array()
    else:
        return from_array(seed)
