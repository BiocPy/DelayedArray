# Changelog

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
