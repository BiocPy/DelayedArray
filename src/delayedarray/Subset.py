from typing import Tuple, Sequence
from collections import namedtuple
from copy import deepcopy
from numpy import dtype, array, ndarray, ix_
from bisect import bisect_left

from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .utils import sanitize_single_index


# Check if the subset is already unique and/or sorted. If it is, we can optimize
# some of the downstream procedures.
def _is_unique_and_sorted(subset):
    is_sorted = True
    for i in range(1, len(subset)):
        if subset[i] < subset[i-1]:
            is_sorted = False
            break

    is_unique = True
    if is_sorted:
        for i in range(1, len(subset)):
            if subset[i] == subset[i-1]:
                is_unique = False
                break
    else:
        occurred = set()
        for s in subset:
            if s in occurred:
                is_unique = False
                break
            occurred.add(s)

    return is_sorted, is_unique


# Converts the subset into a sorted and unique sequence for use in
# extract_*_array(), while also creating a mapping that expands to the 
# original subset, i.e., given 'x, y = _normalize_subset(z, ...)',
# we are guaranteed that 'z == x[y]'.
def _normalize_subset(subset, is_sorted, is_unique):
    if is_unique:
        if is_sorted:
            return (subset, range(len(subset)))
        else:
            subsorted = list(set(subset))
            subsorted.sort()

            new_indices = {}
            for i in range(len(subsorted)):
                new_indices[subsorted[i]] = i

            mapping = []
            for s in subset:
                mapping.append(new_indices[s])

            return (subsorted, mapping)
    else:
        if is_sorted:
            subuniq = []
            mapping = []
            if len(subset):
                subuniq.append(subset[0])
                mapping.append(0)

            for i in range(1, len(subset)):
                if subset[i-1] < subset[i]:
                    subuniq.append(subset[i])
                mapping.append(len(subuniq) - 1)

            return (subuniq, mapping)
        else:
            subsorted = list(set(subset))
            subsorted.sort()

            converter = {}
            for i in range(len(subsorted)):
                converter[subsorted[i]] = i

            mapping = []
            for s in subset:
                mapping.append(converter[s])

            return (subsorted, mapping)


def is_subset_noop(idx, full):
    if len(idx) != full:
        return False
    for i in range(full):
        if idx[i] != i:
            return False
    return True


class Subset:
    def __init__(self, seed, subset):
        self._seed = seed
        if len(subset) != len(seed.shape):
            raise ValueError("Dimensionality of 'seed' and 'subset' should be the same.")

        self._is_unique = []
        self._is_sorted = []
        self._subset = []
        self._full_normalized_subset = []
        self._full_subset_mapping = []

        for i in range(len(seed.shape)):
            current = sanitize_single_index(subset[i], seed.shape[i])
            self._subset.append(current)

            s, u = _is_unique_and_sorted(current)
            self._is_unique.append(u)
            self._is_sorted.append(s)

            n, m = _normalize_subset(subset[i], is_sorted=s, is_unique=u)
            self._full_normalized_subset.append(n)
            self._full_subset_mapping.append(m)

        self._shape = (*[len(x) for x in self._subset],)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> dtype:
        return self.seed.dtype


# Combines the subset in the class instance with the indexing request in the
# extract_*_array call. This will fall back to the full normalized subset 
# if it figures out that we're dealing with a no-op index; otherwise it
# will take an indexed slice of the subset and re-normalize it.
def _indexed_subsets(x: Subset, idx: Tuple[Sequence, ...]) -> Tuple[list, list]:
    out_sub = []
    out_map = []

    for i in range(len(idx)):
        curshape = x.shape[i]
        curidx = sanitize_single_index(idx[i], curshape)

        if is_subset_noop(curidx, curshape):
            out_sub.append(x._full_normalized_subset[i])
            out_map.append(x._full_subset_mapping[i])
        else:
            subsub = []
            cursub = x._subset[i]
            for j in curidx:
                subsub.append(cursub[j])

            n, m = _normalize_subset(subsub, is_sorted=x._is_sorted[i], is_unique=x._is_unique[i])
            out_sub.append(n)
            out_map.append(m)

    return out_sub, out_map


@extract_dense_array.register
def _extract_dense_array_Subset(x: Subset, idx: Tuple[Sequence, ...]) -> ndarray:
    subsets, mappings = _indexed_subsets(x, idx)
    compact = extract_dense_array(x._seed, (*subsets,))
    return compact[ix_(*mappings)]


LastIndexInfo = namedtuple('LastIndexInfo', ["first", "last", "inverted", "full", "is_sorted", "is_unique"])


def _inflate_sparse_vector(indices, values, last_info):
    if last_info is None:
        return (indices, values)

    new_indices = []
    new_values = []

    start_pos = 0
    if last_info.first:
        start_pos = bisect_left(indices, last_info.first)

    end_pos = len(indices)
    if last_info.last != last_info.full:
        end_pos = bisect_left(indices, last_info.last, lo=start_pos, hi=end_pos)

    if last_info.is_unique:
        for i in range(start_pos, end_pos):
            remapped = last_info.inverted[indices[i]]
            if remapped is not None:
                new_indices.append(remapped)
                new_values.append(values[i])
    else:
        for i in range(start_pos, end_pos):
            remapped = last_info.inverted[indices[i]]
            if remapped is not None:
                new_indices += remapped
                new_values += [values[i]] * len(remapped)

    if not last_info.is_sorted:
        for i in range(1, len(new_indices)):
            if new_indices[i] < new_indices[i - 1]:
                new_indices, new_values = zip(*sorted(zip(new_indices, new_values)))
                break

    if len(new_indices):
        return array(new_indices, dtype=indices.dtype), array(new_values, dtype=values.dtype)
    else:
        return None


def _recursive_inflater(contents, dim, ndim, mappings, last_info):
    present = False
    replacement = []
    seen = {}

    if dim == ndim - 2:
        for i in mappings[dim]:
            if i in seen: # speed up matters if we've got duplicated indices.
                replacement.append(deepcopy(seen[i]))
                continue
    
            latest = contents[i]
            if latest is not None:
                latest = _inflate_sparse_vector(latest[0], latest[1], last_info)

            replacement.append(latest)
            seen[i] = latest
    else:
        for i in mappings[dim]:
            if i in seen: # speed up matters if we've got duplicated indices.
                replacement.append(deepcopy(seen[i]))
                continue

            latest = contents[i]
            if latest is not None:
                latest = _recursive_inflater(latest, dim + 1, ndim, mappings, last_info)

            replacement.append(latest)
            seen[i] = latest

    for r in replacement:
        if r is not None:
            return replacement

    return None

        
@extract_sparse_array.register
def _extract_sparse_array_Subset(x: Subset, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    subsets, mappings = _indexed_subsets(x, idx)
    compact = extract_sparse_array(x._seed, (*subsets,))

    last_subset = subsets[-1]
    last_mapping = mappings[-1]
    inverted = None
    last_unique = x._is_unique[-1]
    last_sorted = x._is_sorted[-1]
    last_shape = len(last_subset)

    if last_unique:
        is_consecutive = False
        if last_sorted:
            is_consecutive = is_subset_noop(last_mapping, last_shape)

        if not is_consecutive:
            inverted = [None] * last_shape
            for i in range(len(last_mapping)):
                inverted[last_mapping[i]] = i

    else:
        inverted = [None] * last_shape
        for i in range(len(last_mapping)):
            m = last_mapping[i]
            if inverted[m] is None:
                inverted[m] = [i]
            else:
                inverted[m].append(i)

    if inverted is not None:
        last_info = LastIndexInfo(
            first=min(last_mapping), 
            last=max(last_mapping) + 1,
            full=last_shape,
            inverted=inverted,
            is_sorted=last_sorted,
            is_unique=last_unique
        )
    else:
        last_info = None

    if isinstance(compact._contents, list):
        compact._contents = _recursive_inflater(compact._contents, 0, len(x.shape), mappings, last_info)
    elif compact._contents is not None:
        compact._contents = _inflate_sparse_vector(compact._contents[0], compact._contents[1], last_info)

    final_shape = [len(m) for m in mappings]
    compact._shape = (*final_shape,)
    return compact


@is_sparse.register
def _is_sparse_Subset(x: Subset) -> bool:
    return is_sparse(x._seed)
