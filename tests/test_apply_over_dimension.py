import numpy as np
import delayedarray as da
import math
import pytest

from utils import simulate_SparseNdarray


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
    assert da.choose_block_size_for_1d_iteration(x, 0, buffer_size=800) == 10
    assert da.choose_block_size_for_1d_iteration(x, 1, buffer_size=800) == 1

    # No buffer_size.
    assert da.choose_block_size_for_1d_iteration(x, 0, buffer_size=0) == 1
    assert da.choose_block_size_for_1d_iteration(x, 1, buffer_size=0) == 1

    # Behaves correctly with empty objects.
    empty = np.random.rand(100, 0)
    assert da.choose_block_size_for_1d_iteration(empty, 0) == 100
    assert da.choose_block_size_for_1d_iteration(empty, 1) == 1

    # Making a slightly more complex situation.
    x = _ChunkyBoi((100, 200), (20, 25))
    assert da.choose_block_size_for_1d_iteration(x, 0, buffer_size=4000) == 2
    assert da.choose_block_size_for_1d_iteration(x, 1, buffer_size=4000) == 5
    assert da.choose_block_size_for_1d_iteration(x, 0, buffer_size=40000) == 20
    assert da.choose_block_size_for_1d_iteration(x, 1, buffer_size=40000) == 50


def _dense_sum(position, block):
    ss = block.sum()
    if ss is np.ma.masked:
        ss = 0
    return position, ss


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_apply_over_dimension_dense(mask_rate):
    x = np.ndarray([100, 200])
    counter = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = counter
            counter += 1

    if mask_rate:
        mask = np.random.rand(*x.shape) < mask_rate 
        x = np.ma.MaskedArray(x, mask=mask)

    output = da.apply_over_dimension(x, 0, _dense_sum, block_size=3)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert x.sum() == sum(y[1] for y in output)
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, _dense_sum, block_size=7)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert x.sum() == sum(y[1] for y in output)
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)

    # Same results with the default.
    output = da.apply_over_dimension(x, 0, _dense_sum)
    assert x.sum() == sum(y[1] for y in output)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
def test_apply_over_dimension_sparse(mask_rate):
    x = simulate_SparseNdarray((100, 200), mask_rate=mask_rate)

    expected = 0
    for v in x.contents:
        if v is not None:
            subtotal = v[1].sum()
            if subtotal is not np.ma.masked:
                expected += subtotal

    output = da.apply_over_dimension(x, 0, _dense_sum, block_size=3)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert np.allclose(expected, sum(y[1] for y in output))
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, _dense_sum, block_size=7)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert np.allclose(expected, sum(y[1] for y in output))
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)

    # Now activating sparse mode.
    def _sparse_sum(position, block):
        assert isinstance(block, da.SparseNdarray)
        total = 0 
        if block.contents is not None:
            for v in block.contents:
                if v is not None:
                    subtotal = v[1].sum()
                    if subtotal is not np.ma.masked:
                        total += subtotal
        return position, total

    output = da.apply_over_dimension(x, 0, _sparse_sum, block_size=3, allow_sparse=True)
    assert len(output) == math.ceil(x.shape[0]/3)
    assert np.allclose(expected, sum(y[1] for y in output))
    assert output[0][0] == (0, 3)
    assert output[-1][0] == (99, 100)

    output = da.apply_over_dimension(x, 1, _sparse_sum, block_size=7, allow_sparse=True)
    assert len(output) == math.ceil(x.shape[1]/7)
    assert np.allclose(expected, sum(y[1] for y in output))
    assert output[0][0] == (0, 7)
    assert output[-1][0] == (196, 200)


def test_apply_over_dimension_empty():
    x = np.ndarray([100, 0])
    output = da.apply_over_dimension(x, 0, _dense_sum)
    assert len(output) == 1

    output = da.apply_over_dimension(x, 1, _dense_sum)
    assert len(output) == 0
