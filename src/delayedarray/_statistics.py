from typing import List, Tuple
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
