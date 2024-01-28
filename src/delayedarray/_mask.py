from typing import Tuple, Sequence, List
import numpy


# Despite the simplicity of this function, we still write a wrapper around it
# so we can forbid the naked use of array() and ndarray() throughout the rest
# of the package. This encourages proper consideration of masking and ensures
# that we don't accidentally drop masks by using a naked array() somewhere.
def _allocate_unmasked_ndarray(shape: Tuple[int, ...], dtype: numpy.dtype) -> numpy.ndarray:
    return numpy.ndarray(shape, dtype=dtype)


def _allocate_maybe_masked_ndarray(shape: Tuple[int, ...], dtype: numpy.dtype, masked: bool) -> numpy.ndarray:
    output = numpy.ndarray(shape, dtype=dtype)
    if masked:
        output = numpy.ma.MaskedArray(output, mask=False)
    return output


def _convert_to_unmasked_1darray(contents: Sequence, dtype: numpy.dtype) -> numpy.ndarray:
    return numpy.array(contents, dtype=dtype)


def _convert_to_maybe_masked_1darray(contents: Sequence, dtype: numpy.dtype, masked: bool) -> numpy.ndarray:
    output = _allocate_maybe_masked_ndarray((len(contents),), dtype=dtype, masked=masked)
    # This seems to be the only way to safely assign a masked element. Doing
    # 'output[:] = contents' causes some coercion to lose maskingness. 
    for i, y in enumerate(contents):
        output[i] = y
    return output


# Same logic as above; force the rest of the package to make an explicit choice
# between concatenate() and ma.concatenate() to consider masking properly.
def _concatenate_unmasked_ndarrays(x: List[numpy.ndarray], axis: int) -> numpy.ndarray:
    return numpy.concatenate(x, axis=axis)


def _concatenate_maybe_masked_ndarrays(x: List[numpy.ndarray], axis: int, masked: bool) -> numpy.ndarray:
    if masked:
        return numpy.ma.concatenate(x, axis=axis)
    else:
        return numpy.concatenate(x, axis=axis)
