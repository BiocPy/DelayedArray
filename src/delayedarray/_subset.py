from typing import Sequence, Tuple, Union, Callable
from numpy import prod, ndarray, integer, issubdtype, array, ix_, get_printoptions


def _spawn_indices(shape: Tuple[int, ...]) -> Tuple[Sequence[int], ...]:
    return (*[range(s) for s in shape],)


def _is_subset_consecutive(subset: Sequence):
    for s in range(1, len(subset)):
        if subset[s] != subset[s-1]+1:
            return False
    return True


def _is_subset_noop(shape: Tuple[int, ...], subset: Tuple[Sequence, ...]):
    if subset is not None:
        for i, s in enumerate(shape):
            cursub = subset[i]
            if len(cursub) != s:
                return False
            for j in range(s):
                if cursub[j] != j:
                    return False
    return True


def _sanitize_subset(subset: Sequence): 
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


def _getitem_subset_preserves_dimensions(shape: Tuple[int, ...], args: Tuple[Union[slice, Sequence], ...]):
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
        flattened = []
        for d, idx in enumerate(args):
            flattened.append(idx.reshape((prod(idx.shape),)))
        return (*flattened,)

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
        flattened = []
        for d, idx in enumerate(args):
            if isinstance(idx, slice):
                flattened.append(range(*idx.indices(shape[d])))
            else:
                dummy = array(range(shape[d]))[idx]
                flattened.append(dummy)
        return (*flattened,)

    return None


def _getitem_subset_discards_dimensions(x, args: Tuple[Union[slice, Sequence], ...], injected_extract_dense_array: Callable):
    failed = False
    sanitized = []
    remapping = []
    no_remap = 0
    discards = []
    shape = x.shape

    for d, idx in enumerate(args):
        if isinstance(idx, ndarray):
            if len(idx.shape) != 1: 
                raise NotImplementedError("high-dimensional index arrays are not supported yet")

        elif isinstance(idx, slice):
            idx = range(*idx.indices(shape[d]))
        elif not isinstance(idx, Sequence):
            sanitized.append([idx])
            remapping.append([0])
            discards.append(0)
            continue

        san, mapping = _sanitize_subset(idx)
        sanitized.append(san)
        if mapping is None:
            remapping.append(range(shape[d]))
            no_remap += 1
        else:
            remapping.append(mapping)
        discards.append(slice(None))

    out = injected_extract_dense_array(x, sanitized)
    if no_remap < len(shape):
        out = out[ix_(*remapping,)]
    return out[(*discards,)]


def _repr_subset(shape: Tuple[int, ...]):
    total = 1
    for s in shape:
        total *= s

    if total > get_printoptions()["threshold"]:
        ndims = len(shape)
        indices = []
        edge_size = get_printoptions()["edgeitems"]
        for d in range(ndims):
            extent = shape[d]
            if extent > edge_size * 2:
                indices.append(list(range(edge_size + 1)) + list(range(extent - edge_size, extent)))
            else:
                indices.append(range(extent))
    else:
        indices = [range(d) for d in shape]

    return (*indices,)
