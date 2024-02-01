import numpy as np
import delayedarray as da
import math
import pytest

from utils import simulate_SparseNdarray


def _dense_sum(position, block):
    ss = block.sum()
    if ss is np.ma.masked:
        ss = 0
    return position, ss


@pytest.mark.parametrize("mask_rate", [0, 0.2])
@pytest.mark.parametrize("buffer_size", [100, 1000, 10000])
def test_apply_over_dimension_dense(mask_rate, buffer_size):
    x = np.ndarray([100, 200])
    counter = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = counter
            counter += 1

    if mask_rate:
        mask = np.random.rand(*x.shape) < mask_rate 
        x = np.ma.MaskedArray(x, mask=mask)

    output = da.apply_over_dimension(x, 0, _dense_sum, buffer_size=buffer_size)
    assert x.sum() == sum(y[1] for y in output)

    output = da.apply_over_dimension(x, 1, _dense_sum, buffer_size=buffer_size)
    assert x.sum() == sum(y[1] for y in output)


@pytest.mark.parametrize("mask_rate", [0, 0.2])
@pytest.mark.parametrize("buffer_size", [100, 1000, 10000])
def test_apply_over_dimension_sparse(mask_rate, buffer_size):
    x = simulate_SparseNdarray((100, 200), mask_rate=mask_rate)

    expected = 0
    for v in x.contents:
        if v is not None:
            subtotal = v[1].sum()
            if subtotal is not np.ma.masked:
                expected += subtotal

    output = da.apply_over_dimension(x, 0, _dense_sum, buffer_size=buffer_size)
    assert np.allclose(expected, sum(y[1] for y in output))

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

    output = da.apply_over_dimension(x, 0, _sparse_sum, allow_sparse=True, buffer_size=buffer_size)
    assert np.allclose(expected, sum(y[1] for y in output))


def test_apply_over_dimension_empty():
    x = np.ndarray([100, 0])
    output = da.apply_over_dimension(x, 0, _dense_sum)
    assert len(output) == 0 

    output = da.apply_over_dimension(x, 1, _dense_sum)
    assert len(output) == 0
