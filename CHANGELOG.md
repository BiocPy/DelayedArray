# Changelog

## Version 0.6.0

- chore: Remove Python 3.8 (EOL).
- precommit: Replace docformatter with ruff's formatter.

## Version 0.5.3

- Coerce `numpy.generic` instances to `dtype` in the `SparseNdarray` constructor, for consistency with NumPy functions.
- Avoid cast to float when operating on uint64 indices in a `SparseNdarray`.

## Version 0.5.2

- Support `dtype=` and `copy=` arguments in `__array__()`, as required by NumPy v2.
- Added getters for the `RegularTick` properties, for convenience.

## Version 0.5.1

- Add support for more numpy functions to dense and sparse `DelayedArrays`.

## Version 0.5.0

- Switch to `Grid` classes to control how iteration is performed. This supercedes the previous `chunk_shape()` generic.

## Version 0.4.0

- Added a `buffer_size=` option to the `apply*` functions.
- Provide converter generics to SciPy classes, with methods for `SparseNdarray` and sparse `DelayedArray`s.
- Converted th `to_*_array()` functions into generics.
- Correctly handle zero-extent arrays in the `apply*` functions.
- Added methods to compute basic statistics (e.g., sum, variance) from a `SparseNdarray`.
- Added a copy method for `SparseNdarray`.
- Use block processing to compute basic statistics from a `DelayedArray`.
- Do not require Fortran storage order from `extract_dense_array()`.
- The `subset=` argument is now mandatory in `extract_*_array()` calls.
- Bugfix to respect changes in index dtype upon `SparseNdarray` concatenation.

## Version 0.3.5

- Support masking throughout the various operations and methods. 

## Version 0.3.4

- Fixes to maintain support for Python 3.8.

## Version 0.3.3

- Support the delayed unary `logical_not` operation.
- Added utilities like `choose_block_shape_for_iteration()` and `apply_over_blocks()` to iterate over the dimensions or blocks.

## Version 0.3.2

- Bugfix for row-wise combining of 2-dimensional `SparseNdarray`s.

## Version 0.3.1

- Added a `wrap()` method that dispatches to different `DelayedArray` subclasses based on the seed.

## Version 0.3.0

- Replace the `__DelayedArray` methods with generics, for easier extensibility to classes outside of our control.
- Restored the `SparseNdarray` class, to provide everyone with a consistent type during sparse operations.
- Adapted `extract_array()` into the `extract_dense_array()` generic, which now always returns a (Fortran-order) NumPy array.
- Added the `extract_sparse_array()` generic, which always returns a `SparseNdarray` object for sparse arrays.
- Added the `is_sparse()` generic, which determines whether an object is sparse.
- Minor fixes to the `repr()` method for `DelayedArray` objects.
- **scipy** is no longer required for installation but will be used if available.

## Version 0.2.3

- Added a `chunk_shape()` generic to identify the "best" direction for iterating over the matrix.
- Added an easy way to compute iteration widths over the desired dimension.
- Corrected the reported `dtype` from a delayed `Cast`.

## Version 0.0.3

- separate dense and sparse matrix classes

## Version 0.0.1

- initial classes for H5 backed matrices
