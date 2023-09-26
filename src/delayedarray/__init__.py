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
from .utils import create_dask_array, chunk_shape, guess_iteration_block_size, is_sparse
from .extract_dense_array import extract_dense_array
from .extract_sparse_array import extract_sparse_array
from .is_pristine import is_pristine
from .SparseNdarray import SparseNdarray
