import numpy as np
import delayedarray as da
import math
import scipy


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


@da.chunk_shape.register
def chunk_shape_ChunkyBoi(x: _ChunkyBoi):
    return x._chunks


def test_choose_block_size_for_1d_iteration():
    x = np.random.rand(100, 10)
    assert da.choose_block_size_for_1d_iteration(x, 0, memory=800) == 10
    assert da.choose_block_size_for_1d_iteration(x, 1, memory=800) == 1

    # No memory.
    assert da.choose_block_size_for_1d_iteration(x, 0, memory=0) == 1
    assert da.choose_block_size_for_1d_iteration(x, 1, memory=0) == 1

    # Making a slightly more complex situation.
    x = _ChunkyBoi((100, 200), (20, 25))
    assert da.choose_block_size_for_1d_iteration(x, 0, memory=4000) == 2
    assert da.choose_block_size_for_1d_iteration(x, 1, memory=4000) == 5
    assert da.choose_block_size_for_1d_iteration(x, 0, memory=40000) == 20
    assert da.choose_block_size_for_1d_iteration(x, 1, memory=40000) == 50


def test_apply_over_dimension_dense():
    x = np.ndarray([100, 200])
    counter = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = counter
            counter += 1

    def dense_sum(position, block):
        return position, block.sum()

    output = da.apply_over_dimension(x, 0, dense_sum, block_size=3)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert x.sum() == sum(y[1] for y in output)
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, dense_sum, block_size=7)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert x.sum() == sum(y[1] for y in output)
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)

    # Same results with the default.
    output = da.apply_over_dimension(x, 0, dense_sum)
    assert x.sum() == sum(y[1] for y in output)


def test_apply_over_dimension_sparse():
    x = scipy.sparse.random(100, 200, 0.2).tocsc()

    def dense_sum(position, block):
        return position, block.sum()

    output = da.apply_over_dimension(x, 0, dense_sum, block_size=3)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert np.allclose(x.sum(), sum(y[1] for y in output))
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, dense_sum, block_size=7)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert np.allclose(x.sum(), sum(y[1] for y in output))
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)

    # Now activating sparse mode.
    def sparse_sum(position, block):
        assert isinstance(block, da.SparseNdarray)
        total = 0 
        for v in block.contents:
            if v is not None:
                total += v[1].sum()
        return position, total

    output = da.apply_over_dimension(x, 0, sparse_sum, block_size=3, allow_sparse=True)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert np.allclose(x.sum(), sum(y[1] for y in output))
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, sparse_sum, block_size=7, allow_sparse=True)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert np.allclose(x.sum(), sum(y[1] for y in output))
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)
