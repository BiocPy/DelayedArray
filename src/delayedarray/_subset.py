def _spawn_indices(shape):
    return [range(s) for s in shape]


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
