import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .BinaryIsometricOp import BinaryIsometricOp
from .Cast import Cast
from .Combine import Combine
from .DelayedOp import DelayedOp
from .DelayedArray import DelayedArray
from .Round import Round
from .Subset import Subset
from .Transpose import Transpose
from .UnaryIsometricOpSimple import UnaryIsometricOpSimple
from .UnaryIsometricOpWithArgs import UnaryIsometricOpWithArgs
from .SparseNdarray import SparseNdarray

from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .to_dense_array import to_dense_array
from .to_sparse_array import to_sparse_array
from .to_scipy_sparse_matrix import *

from .create_dask_array import create_dask_array
from .is_sparse import is_sparse
from .is_masked import is_masked
from .chunk_shape import chunk_shape
from .is_pristine import is_pristine
from .apply_over_dimension import apply_over_dimension, choose_block_size_for_1d_iteration, guess_iteration_block_size
from .apply_over_blocks import apply_over_blocks, choose_block_shape_for_iteration
from .wrap import wrap
