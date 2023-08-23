import delayedarray
import numpy
from utils import *


def test_UnaryIsometricOpSimple_dense():
    test_shape = (10, 15, 20)
    y = numpy.random.rand(*test_shape)
    full_index = (slice(None), slice(None), slice(None))

    op = delayedarray.UnaryIsometricOpSimple(y, "exp")
    assert not delayedarray.is_sparse(op)
    assert (delayedarray.extract_dense_array(op, full_index) == numpy.exp(y)).all()

    contents = mock_SparseNdarray_contents(test_shape)
    ys = delayedarray.SparseNdarray(test_shape, contents)
    ops = delayedarray.UnaryIsometricOpSimple(ys, "exp")
    assert not delayedarray.is_sparse(ops)
    assert (
        delayedarray.extract_dense_array(ops, full_index)
        == numpy.exp(delayedarray.extract_dense_array(ys, full_index))
    ).all()

    # Works with a slice.
    sub_index = (slice(1, 9), slice(2, 14), slice(0, 20, 2))
    assert (
        delayedarray.extract_dense_array(op, sub_index)
        == numpy.exp(y[(..., *sub_index)])
    ).all()
    assert (
        delayedarray.extract_dense_array(ops, sub_index)
        == numpy.exp(delayedarray.extract_dense_array(ys, sub_index))
    ).all()


def test_UnaryIsometricOpSimple_sparse():
    test_shape = (50, 20)
    y = numpy.random.rand(*test_shape)
    full_index = (slice(None), slice(None))

    op = delayedarray.UnaryIsometricOpSimple(y, "expm1")
    assert not delayedarray.is_sparse(op)
    assert (delayedarray.extract_dense_array(op, full_index) == numpy.expm1(y)).all()

    contents = mock_SparseNdarray_contents(test_shape)
    ys = delayedarray.SparseNdarray(test_shape, contents)
    ops = delayedarray.UnaryIsometricOpSimple(ys, "abs")
    assert delayedarray.is_sparse(ops)
    assert (
        delayedarray.extract_dense_array(ops, full_index)
        == numpy.abs(delayedarray.extract_dense_array(ys, full_index))
    ).all()

    # Works with a slice.
    sub_index = (slice(10, 40, 3), slice(2, 18))
    assert (
        delayedarray.extract_dense_array(op, sub_index)
        == numpy.expm1(y[(..., *sub_index)])
    ).all()
    assert (
        delayedarray.extract_dense_array(ops, sub_index)
        == numpy.abs(delayedarray.extract_dense_array(ys, sub_index))
    ).all()


def test_UnaryIsometricOpSimple_int_promotion():
    test_shape = (20, 10)
    contents = mock_SparseNdarray_contents(test_shape, density1=0)
    for i in range(len(contents)):
        if contents[i] is not None:
            contents[i] = (contents[i][0], (contents[i][1] * 10).astype(numpy.int32))

    y = delayedarray.SparseNdarray(test_shape, contents)
    assert y.dtype == numpy.int32
    full_index = (slice(None), slice(None))

    op = delayedarray.UnaryIsometricOpSimple(y, "sin")
    assert delayedarray.is_sparse(op)
    assert op.dtype == numpy.float64  # correctly promoted

    out = delayedarray.extract_dense_array(op, full_index)
    assert out.dtype == numpy.float64
    ref = numpy.sin(delayedarray.extract_dense_array(y, full_index))
    assert (out == ref).all()

    spout = delayedarray.extract_sparse_array(op, full_index)
    assert spout.dtype == numpy.float64
    assert (delayedarray.extract_dense_array(spout, full_index) == ref).all()
