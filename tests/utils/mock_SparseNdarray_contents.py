import random
import numpy

def mock_SparseNdarray_contents(shape, density1 = 0.5, density2 = 0.5):
    if len(shape) == 1:
        new_indices = []
        new_values = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density2:
                new_indices.append(i)
                new_values.append(random.gauss(0, 1))
        if len(new_values):
            return numpy.array(new_indices), numpy.array(new_values)
        else:
            return None
    else:
        new_content = []
        for i in range(shape[0]):
            if random.uniform(0, 1) < density1:
                new_content.append(None)
            else:
                new_content.append(mock_SparseNdarray_contents(shape[1:], density1=density1, density2=density2))
        return new_content
