import numpy


def _allocate_ndarray_with_parameters(shape: Tuple[int, ...], dtype: numpy.dtype, masked: bool):
    output = numpy.ndarray(shape, dtype=dtype)
    if masked:
        output = numpy.ma.MaskedArray(output, mask=False)
    return output


def _allocate_ndarray_with_template(template: numpy.ndarray):
    return _allocate_ndarray_with_template(template.shape, template.dtype, numpy.ma.is_masked(template))


def _create_ndarray_with_parameters(contents: Sequence, dtype: numpy.dtype, masked: bool):
    output = _allocate_ndarray_with_parameters((len(contents),), dtype, masked)
    for i, y in enumerate(contents):
        output[i] = y
    return output


def _create_ndarray_with_template(contents: Sequence, template: numpy.ndarray):
    output = _allocate_ndarray_with_parameters((len(contents),), template.dtype, numpy.ma.is_masked(template))
    for i, y in enumerate(contents):
        output[i] = y
    return output


def _create_possibly_masked_ndarray_with_parameters(contents: Sequence, dtype: numpy.dtype):
    output = _allocate_ndarray_with_parameters((len(contents),), dtype, any(numpy.ma.is_masked(y) for y in contents))
    for i, y in enumerate(contents):
        output[i] = y
    return output
