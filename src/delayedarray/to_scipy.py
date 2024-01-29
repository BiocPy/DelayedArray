import numpy
from typing import Any
from biocutils.package_utils import is_package_installed

from .SparseNdarray import SparseNdarray
from .extract_sparse_array import to_sparse_array


if is_package_installed("scipy"):
    import scipy.sparse


    def to_scipy_csc_matrix(x: Any) -> scipy.sparse.csc_matrix:
        """
        Convert a 2-dimensional ``DelayedArray`` or ``SparseNdarray`` into a
        SciPy compressed sparse column (CSC) matrix.

        Args:
            x:
                Input matrix where :py:func:`~delayedarray.is_sparse.is_sparse`
                returns True and :py:func:`~delayedarray.is_sparse.is_masked`
                returns False.

        Returns:
            A CSC matrix with the contents of ``x``.
        """
        # One might think that we could be more memory-efficient by doing block
        # processing. However, there is no advantage from doing so as we eventually
        # need to hold all the blocks in memory before concatenation. We'd only
        # avoid this if we did two passes; one to collect the total size for
        # allocation, and another to actually fill the vectors; not good, so we
        # just forget about it and load it all into memory up-front.
        if not isinstance(x, SparseNdarray):
            x = to_sparse_array(x)

        all_indptrs = numpy.zeros(x.shape[1] + 1, dtype=numpy.uint64)
        if x.contents is not None:
            all_indices = []
            all_values = []
            counter = 0
            for i, y in enumerate(x.contents):
                if y is not None:
                    counter += len(y[0])
                    all_indices.append(y[0])
                    all_values.append(y[1])
                all_indptrs[i + 1] = counter
            all_indices = numpy.concatenate(all_indices)
            all_values = numpy.concatenate(all_values)
        else:
            all_indices = numpy.zeros(0, dtype=x.index_dtype)
            all_values = numpy.zeros(0, dtype=x.dtype)

        return scipy.sparse.csc_matrix((all_values, all_indices, all_indptrs), shape=x.shape)


    def to_scipy_csr_matrix(x: Any) -> scipy.sparse.csr_matrix:
        """
        Convert a 2-dimensional ``DelayedArray`` or ``SparseNdarray`` into a
        SciPy compressed sparse row (CSR) matrix.

        Args:
            x:
                Input matrix where :py:func:`~delayedarray.is_sparse.is_sparse`
                returns True and :py:func:`~delayedarray.is_sparse.is_masked`
                returns False.

        Returns:
            A CSR matrix with the contents of ``x``.
        """
        # Same logic as above; block processing just ends up reading the entire
        # thing into memory before forming the full arrays, so we just load it
        # all in to start with and save ourselves the trouble.
        if not isinstance(x, SparseNdarray):
            x = to_sparse_array(x)

        all_indptrs = numpy.zeros(x.shape[0] + 1, dtype=numpy.uint64)
        if x.contents is not None:
            # First pass (in memory) to obtain the total sizes.
            for i, y in enumerate(x.contents):
                if y is not None:
                    for ix in y[0]:
                        all_indptrs[ix + 1] += 1

            for i in range(1, len(all_indptrs)):
                all_indptrs[i] += all_indptrs[i - 1]
            all_indices = numpy.ndarray(all_indptrs[-1], dtype=x.index_dtype)
            all_values = numpy.ndarray(all_indptrs[-1], dtype=x.dtype)

            # Second pass to fill the allocations that we just made.
            offsets = all_indptrs.copy()
            for i, y in enumerate(x.contents):
                if y is not None:
                    vals = y[1]
                    for j, ix in enumerate(y[0]):
                        o = offsets[ix]
                        all_indices[o] = i
                        all_values[o] = vals[j]
                        offsets[ix] += 1
        else:
            all_indices = numpy.zeros(0, dtype=x.index_dtype)
            all_values = numpy.zeros(0, dtype=x.dtype)

        return scipy.sparse.csr_matrix((all_values, all_indices, all_indptrs), shape=x.shape)


    def to_scipy_coo_matrix(x: Any) -> scipy.sparse.coo_matrix:
        """
        Convert a 2-dimensional ``DelayedArray`` or ``SparseNdarray`` into a
        SciPy sparse coordinate (COO) matrix.

        Args:
            x:
                Input matrix where :py:func:`~delayedarray.is_sparse.is_sparse`
                returns True and :py:func:`~delayedarray.is_sparse.is_masked`
                returns False.

        Returns:
            A COO matrix with the contents of ``x``.
        """
        # Same logic as above.
        if not isinstance(x, SparseNdarray):
            x = to_sparse_array(x)

        if x.contents is not None:
            # First pass (in memory) to obtain the total sizes.
            total_count = 0
            for i, y in enumerate(x.contents):
                if y is not None:
                    total_count += len(y[0])

            all_rows = numpy.ndarray(total_count, dtype=x.index_dtype)
            all_cols = numpy.ndarray(total_count, dtype=numpy.uint64)
            all_values = numpy.ndarray(total_count, dtype=x.dtype)

            # Second pass to fill the allocations that we just made.
            counter = 0
            for i, y in enumerate(x.contents):
                if y is not None:
                    vals = y[1]
                    for j, ix in enumerate(y[0]):
                        all_rows[counter] = ix
                        all_cols[counter] = i
                        all_values[counter] = vals[j]
                        counter += 1
        else:
            all_indices = numpy.zeros(0, dtype=x.index_dtype)
            all_values = numpy.zeros(0, dtype=x.dtype)

        return scipy.sparse.coo_matrix((all_values, (all_rows, all_cols)), shape=x.shape)
