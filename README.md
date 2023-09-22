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

This package implements classes for delayed array operations, mirroring the [Bioconductor package](https://bioconductor.org/packages/DelayedArray) of the same name.
It allows BiocPy-based packages to easily inteoperate with delayed arrays from the Bioconductor ecosystem,
with focus on serialization to/from file with [**chihaya**](https://github.com/ArtifactDB/chihaya)/[**rds2py**](https://github.com/BiocPy/rds2py)
and entry into [**tatami**](https://github.com/tatami-inc/tatami)-compatible C++ libraries via [**mattress**](https://github.com/BiocPy/mattress).

## Installation

This package is published to [PyPI](https://pypi.org/project/delayedarray/) and can be installed via the usual methods:

```shell
pip install delayedarray
```

## Quick start

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
d = delayedarray.DelayedArray(x)
## <100 x 20> DelayedArray object of type 'float64'
## [[0.87165637, 0.37536154, 0.49505459, ..., 0.90147358, 0.13091768,
##   0.7288351 ],
##  [0.06014594, 0.04758512, 0.1932337 , ..., 0.83628993, 0.63886397,
##   0.37175146],
##  [0.86038138, 0.1844154 , 0.45318283, ..., 0.411131  , 0.61720257,
##   0.44831668],
##  ...,
##  [0.2960631 , 0.85775072, 0.83518558, ..., 0.32533032, 0.59257349,
##   0.36232564],
##  [0.7026017 , 0.86221974, 0.42704164, ..., 0.7612019 , 0.58842594,
##   0.51895466],
##  [0.4321901 , 0.29703596, 0.34399029, ..., 0.04685882, 0.20102342,
##   0.05495118]]
```

And then we can use it in a variety of operations.
Each operation just returns a `DelayedArray` with an increasing stack of delayed operations, without evaluating anything or making any copies.

```python
s = d.sum(axis=0)
n = (numpy.log1p(d / s) + 5)[1:5,:]
## <4 x 20> DelayedArray object of type 'float64'
## array([[5.01864954, 5.01248763, 5.00465425, 5.01366904, 5.01444268,
##         5.01740277, 5.00211704, 5.00456718, 5.01170253, 5.00268081,
##         5.00069047, 5.01792154, 5.01174818, 5.007219  , 5.01613611,
##         5.01998141, 5.00359273, 5.00891747, 5.00167042, 5.00480139],
##        [5.01319369, 5.01366843, 5.00259837, 5.01438949, 5.0168967 ,
##         5.0118356 , 5.01468261, 5.00266368, 5.00820377, 5.01519285,
##         5.00880128, 5.01867732, 5.00597971, 5.0132913 , 5.0169869 ,
##         5.02033736, 5.0054349 , 5.01064519, 5.01484268, 5.00933761],
##        [5.01056552, 5.00430873, 5.01554934, 5.01523742, 5.00447682,
##         5.00896808, 5.01702989, 5.00417863, 5.0106902 , 5.01643898,
##         5.00436048, 5.01041755, 5.01358732, 5.01173475, 5.00581787,
##         5.01454487, 5.0097424 , 5.01313867, 5.01227209, 5.01212552],
##        [5.00265869, 5.01460805, 5.00834077, 5.01877699, 5.00009671,
##         5.01027705, 5.00650493, 5.01116854, 5.00582936, 5.00997989,
##         5.00213256, 5.00145715, 5.00797343, 5.01588012, 5.01435549,
##         5.00294226, 5.01381951, 5.01344824, 5.020751  , 5.01294937]])
```

Check out the [documentation](https://biocpy.github.io/DelayedArray/) for more information.

## Extracting data

Users can call `numpy.array()`, to realize the delayed operations into a typical NumPy array for consumption;
or `delayedarray.extract_array()`, to realize the delayed operations while attempting to preserve the original class (e.g., SciPy sparse matrices);
or `delayedarray.create_dask_array()`, to obtain a **dask** array that contains the delayed operations.

```python
simple = numpy.array(n)
type(simple)
## <class 'numpy.ndarray'>

preserved = delayedarray.extract_array(n)
type(preserved)
## <class 'numpy.ndarray'>

# Note: requires installation as 'delayedarray[dask]'.
dasky = delayedarray.create_dask_array(n)
type(dasky)
## <class 'dask.array.core.Array'>
```

Alternatively, users can process a `DelayedArray` by iteratively extracting contiguous blocks on a dimension of interest.
The use of blocks avoids realizing the entire set of delayed operations at once, while reducing overhead from repeated calls to `extract_array` .
For example, to iterate over the rows with 100 MB blocks:

```python
block_size = delayedarray.guess_iteration_block_size(d, dimension=0, memory=1e8)
for start in range(0, d.shape[0], block_size):
    end = min(d.shape[0], start + block_size)
    current = delayedarray.extract_array(d, (range(start, end), range(d.shape[1])))
    # Do something with this block
```

## For developers

Ideally, we would use **dask** directly and avoid creating a set of `DelayedArray` wrapper classes.
We could parse the `HighLevelGraph` objects and retrieve the delayed operations for serialization/reconstruction in other frameworks like R and C++.
Unfortunately, it was tricky to parse the call graph reliably (see the [developer notes](https://biocpy.github.io/DelayedArray/developers.html)).
So, the _real_ purpose of the **DelayedArray** package is to make it easier for Bioconductor developers to inspect the delayed operations.
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
