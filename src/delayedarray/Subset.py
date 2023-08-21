from .interface from extract_dense_array, extract_sparse_array
from copy import deepcopy
from numpy import dtype

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

def _normalize_subset(subset, is_sorted, is_unique):
    if is_unique:
        if is_sorted:
            return (subset, None)
        else:
            new_indices = {}
            for i in range(len(subset)):
                new_indices[subset[i]] = i

            subsorted = deepcopy(subset).sort()
            mapping = []
            for s in subsorted:
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
                if subset[i-1] < sub[i]:
                    subuniq.append(sub[i])
                mapping.append(len(subuniq) - 1)

            return (subuniq, mapping)
        else:
            output = list(set(subset)).sort()
            mappings = {}
            for i in range(len(output)):
                mappings[output[i]] = i

            for s in subset:
                if s in mappings:
                    new_indices[subset[i]] = []
                new_indices[subset[i]].append(i)

            subsorted = list(new_indices.keys()).sort()
            mapping = []
            for s in subsorted:
                mapping.append(new_indices[s])

            return (subsorted, mapping)

class Subset:
    def __init__(self, seed, subset):
        self._seed = seed
        self._subset = subset
        if len(subset) != len(seed.shape):
            raise ValueError("Dimensionality of 'seed' and 'subset' should be the same.")

        self._is_unique = []
        self._is_sorted = []
        self._full_normalized_subset = []
        self._full_subset_mapping = []

        for i in range(len(seed.shape)):
            current = sanitize_single_index(subset[i], seed.shape[i])

            u, s = _is_unique_and_sorted(current)
            self._is_unique.append(u)
            self._is_sorted.append(s)

            n, m = _normalize_subset(subset[i])
            self._full_normalized_subset.append(n)
            self._full_subset_mapping.append(m)

        self._shape = (*[len(x) for x in self._full_normalized_subset],)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> dtype:
        return self.seed.dtype







