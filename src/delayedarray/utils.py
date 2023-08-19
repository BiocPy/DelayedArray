__author__ = "ltla"
__copyright__ = "ltla"
__license__ = "MIT"


def sanitize_single_index(idx, shape):
    if isinstance(idx, slice):
        return range(*idx.indices(shape))
    return idx


def sanitize_indices(idx, shape):
    idx2 = []
    for i in range(len(idx)):
        idx2.append(sanitize_single_index(idx[i], shape[i]))
    return idx2
