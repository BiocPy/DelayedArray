import random
import numpy


def _recursive_compute_reference(contents, at, max_depth, triplets):
    if len(at) == max_depth - 2:
        for i in range(len(contents)):
            if contents[i] is not None:
                idx, val = contents[i]
                for j in range(len(idx)):
                    triplets.append(((*at, i, idx[j]), val[j]))
    else:
        for i in range(len(contents)):
            if contents[i] is not None:
                recursive_compute_reference(contents[i], (*at, i), max_depth, triplets)


def convert_SparseNdarray_to_numpy(contents, shape):
    triplets = []

    if len(shape) == 1:
        idx, val = contents
        for j in range(len(idx)):
            triplets.append(((idx[j],), val[j]))
    else:
        _recursive_compute_reference(contents, (), len(shape), triplets)

    output = numpy.zeros(shape)
    for pos, val in triplets:
        output[(..., *pos)] = val
    return output
