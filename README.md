<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/DelayedArray.svg?branch=main)](https://cirrus-ci.com/github/<USER>/DelayedArray)
[![ReadTheDocs](https://readthedocs.org/projects/DelayedArray/badge/?version=latest)](https://DelayedArray.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/DelayedArray/main.svg)](https://coveralls.io/r/<USER>/DelayedArray)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/DelayedArray.svg)](https://anaconda.org/conda-forge/DelayedArray)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/DelayedArray)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/DelayedArray.svg)](https://pypi.org/project/DelayedArray/)
[![Monthly Downloads](https://pepy.tech/badge/DelayedArray/month)](https://pepy.tech/project/DelayedArray)
![Unit tests](https://github.com/BiocPy/DelayedArray/actions/workflows/pypi-test.yml/badge.svg)

# DelayedArrays, in Python

## Introduction

This package implements classes for delayed array operations, mirroring the [Bioconductor package](https://bioconductor.org/packages/DelayedArray) of the same name.
It allows BiocPy-based packages to easily inteoperate with delayed arrays from the Bioconductor ecosystem,
with focus on serialization to/from file with [**chihaya**](https://github.com/ArtifactDB/chihaya)/[**rds2py**](https://github.com/BiocPy/rds2py)
and entry into [**tatami**](https://github.com/tatami-inc/tatami)-compatible C++ libraries via [**mattress**](https://github.com/BiocPy/mattress).

## Quick start

This package is published to [PyPI](https://pypi.org/project/delayedarray/) and can be installed via the usual methods:

```shell
pip install delayedarray
```

We can create a `DelayedArray` from any object that respects the seed contract,
i.e., has the `shape`/`dtype` properties and supports NumPy slicing.
For example, a typical NumPy array qualifies:

```python
import numpy
x = numpy.random.rand(100, 20)
```

We can wrap this in a `DelayedArray` class:

```python
import delayedarray
d = delayedarray.wrap(x)
## <100 x 20> DelayedArray object of type 'float64'
## [[0.58969193, 0.36342181, 0.03111773, ..., 0.72036247, 0.40297173,
##   0.48654955],
##  [0.96346008, 0.57956493, 0.24247029, ..., 0.49717933, 0.589535  ,
##   0.22806832],
##  [0.61699438, 0.02493104, 0.87487081, ..., 0.44039656, 0.13967301,
##   0.57966883],
##  ...,
##  [0.91583856, 0.94079754, 0.47546576, ..., 0.46866948, 0.87952439,
##   0.81316896],
##  [0.68721591, 0.22789395, 0.51079888, ..., 0.86483248, 0.43933065,
##   0.84304794],
##  [0.47763457, 0.54973367, 0.01159327, ..., 0.47338943, 0.86443755,
##   0.2047926 ]]
```

And then we can use it in a variety of operations.
For example, in genomics, a typical quality control task is to slice the matrix to remove uninteresting features (rows) or samples (columns):

```python
filtered = d[1:100:2,1:8]
filtered.shape
## (50, 7)
```

We then divide by the total sum of each column to compute normalized values between samples.

```python
total = filtered.sum(axis=0)
normalized = filtered / total
normalized.dtype
## dtype('float64')
```

And finally we compute a log-transformation to get some log-normalized values for visualization.

```python
transformed = numpy.log1p(normalized)
transformed[1:5,:]
## <4 x 7> DelayedArray object of type 'float64'
## [[0.03202309, 0.03256592, 0.02281872, ..., 0.03193778, 0.01735653,
##   0.02323571],
##  [0.02668759, 0.0152978 , 0.03818753, ..., 0.00280113, 0.00737041,
##   0.00852137],
##  [0.02125275, 0.01473594, 0.01299548, ..., 0.03092256, 0.01225808,
##   0.0030042 ],
##  [0.02334768, 0.00499055, 0.01804982, ..., 0.00467121, 0.02921965,
##   0.02118322]]
```

Each operation just returns a `DelayedArray` with an increasing stack of delayed operations, without evaluating anything or making any copies.
Check out the [documentation](https://biocpy.github.io/DelayedArray/) for more information.

## Extracting data

### Block processing

A `DelayedArray` is typically used by iteratively extracting blocks into memory for further calculations.
This "block processing" strategy improves memory efficiency by only realizing the delayed operations for a subset of the data.
For example, to iterate over the rows with 100 MB blocks:

```python
block_size = delayedarray.choose_block_size_for_1d_iteration(d, dimension=0, memory=1e8)
block_coords = [ None, range(d.shape[1]) ]

for start in range(0, d.shape[0], block_size):
    end = min(d.shape[0], start + block_size)
    block_coords[0] = range(start, end)
    current = delayedarray.extract_dense_array(d, (*block_coords,))
```

Each call to `extract_dense_array()` yields a NumPy array containing the the specified rows and columns.
If the `DelayedArray` might contain masked values, a NumPy `MaskedArray` is returned instead;
this can be determined by checking whether `is_masked(d)` returns `True`.

The above iteration can be simplified with the `apply_over_dimension()` function, which handles the block coordinate calculations for us.
We could also use the `apply_over_blocks()` function to iterate over arbitrary block shapes, which may be more efficient if the best dimension for iteration is not known.

```python
# To iterate over a single dimension:
delayedarray.apply_over_dimension(
    d,
    dimension=0,
    fun=some_user_supplied_function,
    block_size=block_size,
)

# To iterate over arbitrary blocks.
delayedarray.apply_over_blocks(
    d,
    fun=another_user_supplied_function,
    block_shape=(20, 100),
)
```

### Handling sparse data

If the `DelayedArray` contains sparse data, `is_sparse(d)` will return `True`.
This allows callers to instead use the `extract_sparse_array()` function for block processing:

```python
if delayedarray.is_sparse(d):
    current = delayedarray.extract_sparse_array(d, (*block_coords,))
```

This returns a `SparseNdarray` consisting of a tree of sparse vectors for the specified block.
Users can retrieve the sparse vectors by inspecting the `contents` property of the `SparseNdarray`:

- In the one-dimensional case, this is a tuple of two 1-dimensional NumPy arrays storing data about the non-zero elements.
  The first array contains sorted indices while the secon array contains the associated values.
  If `is_masked(d)` returns `True`, the values will be represented as NumPy `MaskedArray` objects.
- For the two-dimensional case, this is a list of such tuples, with one tuple per column.
  This is roughly analogous to a compressed sparse column matrix.
  An entry of the list may also be `None`, indicating that no non-zero elements are present in that column.
- For higher-dimensionals, the tree is a nested list of lists of tuples.
  Each nesting level corresponds to a dimension; the outermost level contains elements of the last dimension,
  the next nesting level contains elements of the second-last dimension, and so on,
  with the indices in the tuple referring to the first dimension.
  Any list element may be `None` indicating that the corresponding element of the dimension has no non-zero elements.
- In all cases, it is possible for `contents` to be `None`, indicating that there are no non-zero elements in the entire array.

The `apply_over_*` functions can also be instructed to iteratively extract blocks as `SparseNdarray` objects.
This only occurs if the input array is sparse (as specified by `is_sparse`).

```python
# To iterate over a single dimension:
delayedarray.apply_over_dimension(
    d,
    dimension=0,
    fun=some_user_supplied_function,
    block_size=block_size,
    allow_sparse=True,
)
```

### Other coercions

A `DelayedArray` can be converted to a (possibly masked) NumPy array with the `to_dense_array()` function.
Similarly, sparse `DelayedArray`s can be converted to `SparseNdarray`s with the `to_sparse_array()` function.

```python
delayedarray.to_dense_array(d)
delayedarray.to_sparse_array(d)
```

Users can easily convert a 2-dimensional `SparseNdarray` to some of the common SciPy sparse matrix classes downstream calculations.

```python
delayedarray.to_scipy_sparse_matrix(current, "csc")
```

More simply, users can just call `numpy.array()` to realize the delayed operations into a standard NumPy array for consumption.
Note that this discards any masking information so should not be called if `is_masked()` returns `True`.

```python
simple = numpy.array(n)
type(simple)
## <class 'numpy.ndarray'>
```

Users can also call `delayedarray.create_dask_array()`, to obtain a **dask** array that contains the delayed operations:

```python
# Note: requires installation as 'delayedarray[dask]'.
dasky = delayedarray.create_dask_array(n)
type(dasky)
## <class 'dask.array.core.Array'>
```

## Interoperability with other packages 

The general idea is that `DelayedArray`s should be a drop-in replacement for NumPy arrays, at least for [BiocPy](https://github.com/BiocPy) applications.
So, for example, we can stuff the `DelayedArray` inside a `SummarizedExperiment`:

```python
import summarizedexperiment as SE
se = SE.SummarizedExperiment({ "counts": filtered, "lognorm": transformed })
print(se)
## Class SummarizedExperiment with 50 features and 7 samples
##   assays: ['counts', 'lognorm']
##   features: []
##   sample data: []
```

One of the main goals of the **DelayedArray** package is to make it easier for Bioconductor developers to inspect the delayed operations.
(See the [developer notes](https://biocpy.github.io/DelayedArray/developers.html) for some comments on **dask**.)
For example, we can pull out the "seed" object underlying our `DelayedArray` instance:

```python
n.seed
## <delayedarray.Subset.Subset object at 0x11cfbe690>
```

Each layer has its own specific attributes that define the operation, e.g.,

```python
n.seed.subset
## (range(1, 5), range(0, 20))
```

Recursively drilling through the object will eventually reach the underlying array(s):

```python
n.seed.seed.seed.seed.seed
## array([[0.78811524, 0.87684408, 0.56980128, ..., 0.92659988, 0.8716243 ,
##         0.8855508 ],
##        [0.96611119, 0.36928726, 0.30364589, ..., 0.14349135, 0.92921468,
##         0.85097595],
##        [0.98374144, 0.98197003, 0.18126507, ..., 0.5854122 , 0.48733974,
##         0.90127042],
##        ...,
##        [0.05566008, 0.24581195, 0.4092705 , ..., 0.79169303, 0.36982844,
##         0.59997214],
##        [0.81744194, 0.78499666, 0.80940409, ..., 0.65706498, 0.16220355,
##         0.46912681],
##        [0.41896894, 0.58066043, 0.57069833, ..., 0.61640286, 0.47174326,
##         0.7149704 ]])
```

All attributes required to reconstruct a delayed operation are public and considered part of the stable `DelayedArray` interface.

## Developing seeds

Any array-like object can be used as a "seed" in a `DelayedArray` provided it has the following:

- `dtype` and `shape` properties, like those in NumPy arrays.
- a method for the `extract_dense_array()` generic.
- a method for the `is_masked()` generic.

If the object may contain sparse data, it should also implement:

- a method for the `is_sparse()` generic.
- a method for the `extract_sparse_generic()` generic.

It may also be desirable to implement:

- a method for the `chunk_shape()` generic.
- a method for the `create_dask_array()` generic.
- a method for the `wrap()` generic.

Developers are referred to the [documentation for each generic](https://biocpy.github.io/DelayedArray/api/delayedarray.html) for more details.
