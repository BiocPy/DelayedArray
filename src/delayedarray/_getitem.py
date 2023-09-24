from numpy import prod, ndarray, issubdtype, integer, array
from typing import Sequence
import warnings

from .Subset import Subset
from .utils import extract_array, _densify


def _sanitize_getitem(shape, args):
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


def _extract_dense_subarray(seed, shape, args):
    # If we're discarding dimensions, we see if we can do some pre-emptive extraction.
    failed = False
    as_vector = []
    new_args = []
    dim_loss = 0

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
            dim_loss += 1
            continue

        as_vector.append(idx)
        new_args.append(slice(None))

    if not failed:
        # Just using Subset here to avoid having to reproduce the
        # uniquifying/sorting of subsets before extract_array().
        base_seed = extract_array(Subset(seed, (*as_vector,)))
    else:
        base_seed = extract_array(seed)
        new_args = args

    ndim = len(shape)
    try:
        test = base_seed[(..., *new_args)]
        if len(test.shape) != ndim - dim_loss:
            raise ValueError("slicing for " + str(type(base_seed)) + " does not discard dimensions with scalar indices")
    except Exception as e:
        warnings.warn(str(e))
        test = _densify(base_seed)[(..., *new_args)]

    if len(test.shape) == ndim:
        raise NotImplementedError(
            "Oops. Looks like the DelayedArray doesn't correctly handle this combination of index types, but it "
            "probably should. Consider filing an issue in at https://github.com/BiocPy/DelayedArray/issues."
        )

    return test

