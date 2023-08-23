from bisect import bisect_left
from collections import namedtuple
from copy import deepcopy
from typing import Sequence, Tuple

from numpy import array, dtype, ix_, ndarray

from .interface import extract_dense_array, extract_sparse_array, is_sparse
from .SparseNdarray import SparseNdarray
from .utils import sanitize_single_index


# Check if the subset is already unique and/or sorted. If it is, we can optimize
# some of the downstream procedures.
def _is_unique_and_sorted(subset):
    is_sorted = True
    for i in range(1, len(subset)):
        if subset[i] < subset[i - 1]:
            is_sorted = False
            break

    is_unique = True
    if is_sorted:
        for i in range(1, len(subset)):
            if subset[i] == subset[i - 1]:
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
                if subset[i - 1] < subset[i]:
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
    """Delayed subset operation.
    This will slice the array along one or more dimensions, equivalent to the outer product of
    subset indices. The subset can also be used to reduce the dimensionality of the array by
    extracting only one element from one or more dimensions.

    Attributes:
        seed:
            Any object that satisfies the seed contract,
            see :py:class:`~delayedarray.DelayedArray.DelayedArray` for details.

        subset (Tuple[Sequence, ...]):
            Tuple of length equal to the dimensionality of ``seed``, containing the subsetted
            elements for each dimension.

            Each entry may be a vector of integer indices specifying the elements of the
            corresponding dimension to retain, where each integer is non-negative and less than the
            extent of the dimension. Unsorted and/or duplicate indices are allowed.

            Alternatively, each entry may be an integer specifying the single element of interest
            for its dimension. In this case, the result of the subsetting operation will have a
            lower dimensionality than ``seed``.
    """

    def __init__(self, seed, subset: Tuple[Sequence, ...]):
        self._seed = seed
        if len(subset) != len(seed.shape):
            raise ValueError(
                "Dimensionality of 'seed' and 'subset' should be the same."
            )

        self._subset = []

        # These may be shorter than 'self._subset', if dimensionality is lost.
        self._is_unique = []
        self._is_sorted = []
        self._full_normalized_subset = []
        self._full_subset_mapping = []

        final_shape = []
        for i in range(len(seed.shape)):
            cursub = subset[i]
            if isinstance(cursub, int):
                self._subset.append(cursub)
                continue

            cursan = sanitize_single_index(cursub, seed.shape[i])
            self._subset.append(cursan)
            final_shape.append(len(cursan))

            s, u = _is_unique_and_sorted(cursan)
            self._is_unique.append(u)
            self._is_sorted.append(s)

            n, m = _normalize_subset(cursan, is_sorted=s, is_unique=u)
            self._full_normalized_subset.append(n)
            self._full_subset_mapping.append(m)

        self._shape = (*final_shape,)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the array.

        Returns:
            Tuple[int, ...]: Tuple of integers containing the array shape along
            each dimension.
        """
        return self._shape

    @property
    def dtype(self) -> dtype:
        """Type of the array.

        Returns:
            numpy.dtype: Type of the NumPy array containing the values of the non-zero elements.
        """
        return self._seed.dtype


# Combines the subset in the class instance with the indexing request in the
# extract_*_array call. This will fall back to the full normalized subset
# if it figures out that we're dealing with a no-op index; otherwise it
# will take an indexed slice of the subset and re-normalize it.
def _indexed_subsets(x: Subset, idx: Tuple[Sequence, ...]) -> Tuple[list, list, list]:
    out_sub = []
    out_map = []
    not_lost = []

    xshape = x._shape
    raw_subsets = x._subset
    xcount = 0

    for i in range(len(raw_subsets)):
        cursub = raw_subsets[i]
        if isinstance(cursub, int):
            out_sub.append([cursub])
            out_map.append([0])
            continue

        curshape = xshape[xcount]
        curidx = sanitize_single_index(idx[xcount], curshape)

        if is_subset_noop(curidx, curshape):
            out_sub.append(x._full_normalized_subset[xcount])
            out_map.append(x._full_subset_mapping[xcount])
        else:
            subsub = []
            for j in curidx:
                subsub.append(cursub[j])

            n, m = _normalize_subset(
                subsub, is_sorted=x._is_sorted[xcount], is_unique=x._is_unique[xcount]
            )
            out_sub.append(n)
            out_map.append(m)

        not_lost.append(i)
        xcount += 1

    return out_sub, out_map, not_lost


@extract_dense_array.register
def _extract_dense_array_Subset(x: Subset, idx: Tuple[Sequence, ...]) -> ndarray:
    subsets, mappings, not_lost = _indexed_subsets(x, idx)
    compact = extract_dense_array(x._seed, (*subsets,))
    expanded = compact[ix_(*mappings)]

    if len(not_lost) < len(mappings):
        if len(not_lost):
            final_shape = []
            for nl in not_lost:
                final_shape.append(len(mappings[nl]))
            expanded = expanded.reshape(*final_shape)
        else:
            idx = [0] * len(mappings)
            expanded = array(compact[(*idx,)])

    return expanded


LastIndexInfo = namedtuple(
    "LastIndexInfo", ["first", "last", "inverted", "full", "is_sorted", "is_unique"]
)


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
        return array(new_indices, dtype=indices.dtype), array(
            new_values, dtype=values.dtype
        )
    else:
        return None


def _recursive_inflater(contents, dim, mappings, was_lost, last_notlost_dim, last_info):
    seen = {}
    ndim = len(mappings)

    # This clause is necessary to handle cases where the last dimension is lost,
    # meaning that we need to construct an entirely new sparse vector.
    if dim == last_notlost_dim:
        count = 0
        dtype = None
        new_indices = []
        new_values = []

        for i in mappings[dim]:
            if i in seen:
                if seen[i] is not None:
                    new_indices.append(count)
                    new_values.append(seen[i])
            else:
                latest = contents[i]
                while isinstance(latest, list):
                    latest = latest[0]

                if latest is not None:
                    if dtype is not None:
                        dtype = latest.dtype
                    latest = latest[1][0]
                    new_indices.append(count)
                    new_values.append(latest)

                seen[i] = latest

            count += 1

        if len(new_indices) == 0:
            return None
        else:
            return new_indices, array(new_values, dtype=dtype)

    # This clause handles cases where an intermediate dimension is lost,
    # but there is still at least one remaining dimension to be processed.
    if was_lost[dim]:
        latest = contents[0]
        if latest is None:
            return None
        elif dim == ndim - 2:
            return _inflate_sparse_vector(latest[0], latest[1], last_info)
        else:
            return _recursive_inflater(
                latest, dim + 1, mappings, was_lost, last_notlost_dim, last_info
            )

    curmap = mappings[dim]
    replacement = []

    # Now we finally get onto the standard recursion for SparseNdarrays.
    if dim == ndim - 2:
        for i in curmap:
            if i in seen:  # speed up matters if we've got duplicated indices.
                replacement.append(deepcopy(seen[i]))
                continue

            latest = contents[i]
            if latest is not None:
                latest = _inflate_sparse_vector(latest[0], latest[1], last_info)

            replacement.append(latest)
            seen[i] = latest

    else:
        for i in curmap:
            if i in seen:  # speed up matters if we've got duplicated indices.
                replacement.append(deepcopy(seen[i]))
                continue

            latest = contents[i]
            if latest is not None:
                latest = _recursive_inflater(
                    latest, dim + 1, mappings, was_lost, last_notlost_dim, last_info
                )

            replacement.append(latest)
            seen[i] = latest

    for r in replacement:
        if r is not None:
            return replacement

    return None


@extract_sparse_array.register
def _extract_sparse_array_Subset(x: Subset, idx: Tuple[Sequence, ...]) -> SparseNdarray:
    subsets, mappings, not_lost = _indexed_subsets(x, idx)
    compact = extract_sparse_array(x._seed, (*subsets,))

    was_lost = [True] * len(mappings)
    for d in not_lost:
        was_lost[d] = False

    last_info = None
    if len(was_lost) and not was_lost[-1]:
        last_mapping = mappings[-1]
        last_subset = subsets[-1]
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
                is_unique=last_unique,
            )

    # If there are no dimensions, we're dealing with a 0-dimensional
    # SparseNdarray, i.e., a scalar.
    if len(not_lost) == 0:
        contents = compact._contents
        while contents is not None:
            contents = contents[0]
        if contents is not None:
            contents = contents[1]
        return SparseNdarray((), contents, dtype=compact.dtype)

    if isinstance(compact._contents, list):
        compact._contents = _recursive_inflater(
            compact._contents, 0, mappings, was_lost, not_lost[-1], last_info
        )
    elif compact._contents is not None:
        compact._contents = _inflate_sparse_vector(
            compact._contents[0], compact._contents[1], last_info
        )

    final_shape = []
    for d in not_lost:
        final_shape.append(len(mappings[d]))
    compact._shape = (*final_shape,)
    return compact


@is_sparse.register
def _is_sparse_Subset(x: Subset) -> bool:
    return is_sparse(x._seed)
