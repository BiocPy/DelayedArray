from dask.array import from_array

__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def _create_dask_array(seed):
    if hasattr(seed, "as_dask_array"):
        return seed.as_dask_array()
    else:
        return from_array(seed)


def _realize_seed(seed):
    if hasattr(seed, "__DelayedArray_compute__"):
        return seed.__DelayedArray_compute__()
    else:
        return seed
