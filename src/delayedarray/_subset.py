from typing import Sequence, Tuple
from numpy import prod, ndarray, integer, issubdtype, array


def _spawn_indices(shape):
    return [range(s) for s in shape]


def _is_subset_consecutive(subset):
    for s in range(1, len(subset)):
        if subset[s] != subset[s-1]+1:
            return False
    return True


def _is_subset_noop(shape, subset):
    if subset is not None:
        for i, s in enumerate(shape):
            cursub = subset[i]
            if len(cursub) != s:
                return False
            for j in range(s):
                if cursub[j] != j:
                    return False
    return True


def _sanitize_subset(subset):
    okay = True
    for i in range(1, len(subset)):
        if subset[i] <= subset[i - 1]:
            okay = False
            break

    if okay:
        return subset, None

    sortvec = []
    for i, d in enumerate(subset):
        sortvec.append((d, i))
    sortvec.sort()

    san = []
    remap = [None] * len(sortvec)
    last = None
    for d, i in sortvec:
        if last != d:
            san.append(d)
            last = d
        remap[i] = len(san) - 1

    return san, remap


def _sanitize_getitem_subset(shape: Tuple[int], args: Tuple[Union[slice, Sequence]]):
    ndim = len(shape)
    if not isinstance(args, tuple):
        args = [args] + [slice(None)] * (ndim - 1)
    if len(args) < ndim:
        args = list(args) + [slice(None)] * (ndim - len(args))
    elif len(args) > ndim:
        raise ValueError("more indices in 'args' than there are dimensions in 'seed'")

    # Checking if we're preserving the shape via a cross index.
    cross_index = True
    for d, idx in enumerate(args):
        if not isinstance(idx, ndarray) or not issubdtype(idx.dtype, integer) or len(idx.shape) != ndim:
            cross_index = False
            break

        for d2 in range(ndim):
            if d != d2 and idx.shape[d2] != 1:
                cross_index = False
                break

    if cross_index:
        sanitized = []
        for d, idx in enumerate(args):
            sanitized.append(idx.reshape((prod(idx.shape),)))
        return (*sanitized,)

    # Checking if we're preserving the shape via a slice.
    slices = 0
    failed = False
    for d, idx in enumerate(args):
        if isinstance(idx, slice):
            slices += 1
            continue
        elif isinstance(idx, ndarray):
            if len(idx.shape) != 1:
                failed = True
                break
        elif not isinstance(idx, Sequence):
            failed = True
            break

    if not failed and slices >= ndim - 1:
        sanitized = []
        for d, idx in enumerate(args):
            if isinstance(idx, slice):
                sanitized.append(range(*idx.indices(shape[d])))
            else:
                dummy = array(range(shape[d]))[idx]
                sanitized.append(dummy)
        return (*sanitized,)

    return None


def _create_subsets_with_lost_dimension(shape, args):
    # If we're discarding dimensions, we see if we can do some pre-emptive extraction.
    failed = False
    as_vector = []
    new_args = []

    for d, idx in enumerate(args):
        if isinstance(idx, ndarray):
            if len(idx.shape) != 1:
                failed = True
                break
        elif isinstance(idx, slice):
            idx = range(*idx.indices(shape[d]))
        elif not isinstance(idx, Sequence):
            as_vector.append([idx])
            new_args.append(0)
            continue

        as_vector.append(idx)
        new_args.append(slice(None))

    if not failed:
        return True, args
    else:
        return False, args
