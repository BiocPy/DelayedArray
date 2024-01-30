from typing import List, Tuple, Callable, Optional, Union
import numpy


def _find_useful_axes(ndim, axis) -> List[int]:
    output = []
    if axis is not None:
        if isinstance(axis, int):
            if axis < 0:
                axis = ndim + axis
            for i in range(ndim):
                if i != axis:
                    output.append(i)
        else:
            used = set()
            for a in axis:
                if a < 0:
                    a = ndim + a
                used.add(a)
            for i in range(ndim):
                if i not in used:
                    output.append(i)
    return output


def _expected_sample_size(shape: Tuple[int, ...], axes: List[int]) -> int:
    size = 1
    j = 0
    for i, d in enumerate(shape):
        if j == len(axes) or i < axes[j]:
            size *= d
        else:
            j += 1
    return size


def _choose_output_type(dtype: numpy.dtype, preserve_integer: bool) -> numpy.dtype:
    # Mimic numpy.sum's method for choosing the type. 
    if numpy.issubdtype(dtype, numpy.integer):
        if preserve_integer:
            xinfo = numpy.iinfo(dtype)
            if xinfo.kind == "i":
                pinfo = numpy.iinfo(numpy.int_)
                if xinfo.bits < pinfo.bits:
                    dtype = numpy.dtype(numpy.int_)
            else:
                pinfo = numpy.iinfo(numpy.uint)
                if xinfo.bits < pinfo.bits:
                    dtype = numpy.dtype(numpy.uint)
        else:
            dtype = numpy.dtype("float64")
    return dtype


def _allocate_output_array(shape: Tuple[int, ...], axes: List[int], dtype: numpy.dtype) -> numpy.ndarray:
    if len(axes) == 0:
        # Returning a length-1 array to allow for continued use of offsets.
        return numpy.zeros(1, dtype=dtype)
    else:
        # Use Fortran order so that the offsets make sense. 
        shape = [shape[i] for i in axes]
        return numpy.zeros((*shape,), dtype=dtype, order="F")


def _create_offset_multipliers(shape: Tuple[int, ...], axes: List[int]) -> List[int]:
    multipliers = [0] * len(shape)
    sofar = 1
    for a in axes:
        multipliers[a] = sofar
        sofar *= shape[a]
    return multipliers 


def array_sum(x, axis: Optional[Union[int, Tuple[int, ...]]], dtype: Optional[numpy.dtype], reduce_over_x: Callable, masked: bool) -> numpy.ndarray:
    axes = _find_useful_axes(len(x.shape), axis)
    if dtype is None:
        dtype = _choose_output_type(x.dtype, preserve_integer = True)
    output = _allocate_output_array(x.shape, axes, dtype)
    buffer = output.ravel(order="F")

    if masked:
        masked = numpy.zeros(output.shape, dtype=numpy.uint, order="F")
        mask_buffer = masked.ravel(order="F")
        def op(offset, value):
            if value is not numpy.ma.masked:
                buffer[offset] += value
            else:
                mask_buffer[offset] += 1
        reduce_over_x(x, axes, op)
        size = _expected_sample_size(x.shape, axes) 
        output = numpy.ma.MaskedArray(output, mask=(masked == size))
    else:
        def op(offset, value):
            buffer[offset] += value
        reduce_over_x(x, axes, op)

    if len(axes) == 0:
        return output[0]
    else:
        return output


def array_mean(x, axis: Optional[Union[int, Tuple[int, ...]]], dtype: Optional[numpy.dtype], reduce_over_x: Callable, masked: bool) -> numpy.ndarray:
    axes = _find_useful_axes(len(x.shape), axis)
    if dtype is None:
        dtype = _choose_output_type(x.dtype, preserve_integer = False)
    output = _allocate_output_array(x.shape, axes, dtype)
    buffer = output.ravel(order="F")
    size = _expected_sample_size(x.shape, axes) 

    if masked:
        masked = numpy.zeros(output.shape, dtype=numpy.uint, order="F")
        mask_buffer = masked.ravel(order="F")
        def op(offset, value):
            if value is not numpy.ma.masked:
                buffer[offset] += value
            else:
                mask_buffer[offset] += 1
        reduce_over_x(x, axes, op)
        denom = size - masked
        output = numpy.ma.MaskedArray(output, mask=(denom==0))
    else:
        def op(offset, value):
            buffer[offset] += value
        reduce_over_x(x, axes, op)
        denom = size

    output /= denom
    if len(axes) == 0:
        return output[0]
    else:
        return output


def array_var(x, axis: Optional[Union[int, Tuple[int, ...]]], dtype: Optional[numpy.dtype], ddof: int, reduce_over_x: Callable, masked: bool) -> numpy.ndarray:
    axes = _find_useful_axes(len(x.shape), axis)
    if dtype is None:
        dtype = _choose_output_type(x.dtype, preserve_integer = False)
    size = _expected_sample_size(x.shape, axes) 

    # Using Welford's online algorithm.
    sumsq = _allocate_output_array(x.shape, axes, dtype)
    sumsq_buffer = sumsq.ravel(order="F")
    means = numpy.zeros(sumsq.shape, dtype=dtype, order="F")
    means_buffer = means.ravel(order="F")
    counts = numpy.zeros(sumsq.shape, dtype=numpy.int64, order="F")
    counts_buffer = counts.ravel(order="F")

    def raw_op(offset, value):
        counts_buffer[offset] += 1
        delta = value - means_buffer[offset]
        means_buffer[offset] += delta / counts_buffer[offset]
        delta_2 = value - means_buffer[offset]
        sumsq_buffer[offset] += delta * delta_2

    if masked:
        masked = numpy.zeros(sumsq.shape, dtype=numpy.int64, order="F")
        mask_buffer = masked.ravel(order="F")
        def op(offset, value):
            if value is not numpy.ma.masked:
                raw_op(offset, value)
            else:
                mask_buffer[offset] += 1
        reduce_over_x(x, axes, op)
        actual_size = size - masked
        denom = actual_size - ddof
        num_zero = actual_size - counts
        sumsq = numpy.ma.MaskedArray(sumsq, mask = (denom <= 0))
    else:
        reduce_over_x(x, axes, raw_op)
        actual_size = size
        denom = max(0, size - ddof)
        num_zero = size - counts

    old_means = means.copy()
    means *= counts / actual_size
    sumsq += num_zero * (old_means * means)

    sumsq /= denom
    if len(axes) == 0:
        return sumsq[0]
    else:
        return sumsq
