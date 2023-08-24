<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/DelayedArray.svg?branch=main)](https://cirrus-ci.com/github/<USER>/DelayedArray)
[![ReadTheDocs](https://readthedocs.org/projects/DelayedArray/badge/?version=latest)](https://DelayedArray.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/DelayedArray/main.svg)](https://coveralls.io/r/<USER>/DelayedArray)
[![PyPI-Server](https://img.shields.io/pypi/v/DelayedArray.svg)](https://pypi.org/project/DelayedArray/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/DelayedArray.svg)](https://anaconda.org/conda-forge/DelayedArray)
[![Monthly Downloads](https://pepy.tech/badge/DelayedArray/month)](https://pepy.tech/project/DelayedArray)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/DelayedArray)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/DelayedArray.svg)](https://pypi.org/project/DelayedArray/)
![Unit tests](https://github.com/BiocPy/DelayedArray/actions/workflows/pypi-test.yml/badge.svg)

# DelayedArrays, in Python

Pretty much as it says on the tin, we just translated the [R/Bioconductor package of the same name](https://bioconductor.org/packages/DelayedArray) into Python.
This allows BiocPy-based packages to easily inteoperate with delayed arrays from the Bioconductor ecosystem,
with particular focus on serialization to/from file with [**chihaya**](https://github.com/ArtifactDB/chihaya)/[**rds2py**](https://github.com/BiocPy/rds2py)
and use with [**tatami**](https://github.com/tatami-inc/tatami)-compatible C++ libraries via [**mattress**](https://github.com/BiocPy/mattress).
End-users can leverage **DelayedArray** objects to save time and memory by lazily operating on large matrices.
This package is basically a poor man's [**dask**](https://docs.dask.org/en/stable/) that simplifies the class internals for easier cross-language development.

## Installation

This package is published to [PyPI](https://pypi.org/project/delayedarray/) and can be installed via the usual methods:

```shell
pip install delayedarray
```

## Quick start

We can create a `DelayedArray` from any object that respects the seed contract,
i.e., has the `shape`/`dtype` properties and has methods for the `extract_dense_array()`, `is_sparse()` and (optionally) `extract_sparse_array()` generics.
For example, a typical NumPy array qualifies:

```python
import numpy
x = numpy.random.rand(100, 20)
```

We can wrap this in a `DelayedArray` class:

```python
import delayedarray
d = delayedarray.DelayedArray(x)
```

And then we can use it in a variety of operations.
This will just return a `DelayedArray` with an increasing stack of delayed operations, without evaluating anything or making any copies.

```python
n = (numpy.log1p(d / 2) + 5)[1:5,:]
```

Users can then call `numpy.array()` to realize the delayed operations into a typical NumPy array for consumption.

```python
numpy.array(n)
```

Check out the [documentation](https://biocpy.github.io/DelayedArray/) for more info.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
