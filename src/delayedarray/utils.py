def sanitize_indices(idx, shape):
    idx2 = []
    for i in range(len(idx)):
        curidx = idx[i]
        if isinstance(curidx, slice):
            idx2.append(range(*curidx.indices(shape[i])))
        else:
            idx2.append(curidx)
    return idx2
