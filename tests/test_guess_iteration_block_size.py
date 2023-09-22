import numpy as np
import delayedarray as da


class _ChunkyBoi:
    def __init__(self, shape, chunks):
        self._shape = shape
        self._chunks = chunks

    @property
    def dtype(self):
        return np.dtype("float64")

    @property
    def shape(self):
        return self._shape

    def __DelayedArray_chunk__(self):
        return self._chunks


def test_guess_iteration_block_size():
    x = np.random.rand(100, 10)
    assert da.guess_iteration_block_size(x, 0, memory=800) == 10
    assert da.guess_iteration_block_size(x, 1, memory=800) == 1

    # No memory.
    assert da.guess_iteration_block_size(x, 0, memory=0) == 1
    assert da.guess_iteration_block_size(x, 1, memory=0) == 1

    # Making a slightly more complex situation.
    x = _ChunkyBoi((100, 200), (20, 25))
    assert da.guess_iteration_block_size(x, 0, memory=4000) == 2
    assert da.guess_iteration_block_size(x, 1, memory=4000) == 5
    assert da.guess_iteration_block_size(x, 0, memory=40000) == 20
    assert da.guess_iteration_block_size(x, 1, memory=40000) == 50
