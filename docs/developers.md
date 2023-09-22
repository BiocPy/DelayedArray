## Developer notes

As mentioned elsewhere: ideally, we would use **dask** directly, as it supports all of the desired operations and provides a NumPy-compatible interface.
We could then reach into the `.dask` attribute to recover the `HighLevelGraph` and parse it to determine the sequence of delayed operations.
This would save us the trouble of writing the `DelayedArray` classes.

Unfortunately, parsing the call graph exposes BiocPy packages to **dask** internals like the [task specification](https://docs.dask.org/en/latest/spec.html).
While this is not too complicated (usually), it's a risk that could cause our packages to break with different versions of **dask**.
Indeed, from their own [documentation](https://docs.dask.org/en/latest/graphs.html):

> Working directly with dask graphs is rare, though, unless you intend to develop new modules with Dask. Even then, dask.delayed is often a better choice. If you are a core developer, then you should start here.

... which does not give me confidence that third-party developers parsing the call graph will be a priority for the **dask** team when they're thinking about backwards compatibility.
That's fair enough.

Similarly, the deeper parts of the task specification are not well-documented.
For example, each task is represented by a callable object, but these are **dask**-internal objects that do not have clear descriptions of their attributes.
We need to do some further digging to actually get the function involved (`log2`); who knows what the `__dask_blockwise__0` means.

```python
import numpy
import dask.array as da
X = da.from_array(numpy.random.rand(100, 20))
Y = numpy.log2(X)

# Finding the log2 step:
steps = list(Y.dask.get_all_external_keys())
key = None
for x in steps:
    if x[0].startswith("log2-"):
        key = x

key
## ('log2-3999c8c7866c0ec86d68c8ed4da7d784', 0, 0)

# Retrieving the callable:
logstep = Y.dask[key]
logstep[0]
## subgraph_callable-38677753-1f25-4846-ac5c-0fd63671121f

# Actually unpacking the callable:
logstep[0].dsk
{'log2-3999c8c7866c0ec86d68c8ed4da7d784': (<ufunc 'log2'>, '__dask_blockwise__0')}
```

Also, there are occasions when the **dask** does something magical to the call graph, usually when we're operating on multiple arrays.
In the example below, the `all_sums` array is silently split and distributed to the two submatrices
(i.e., the `rechunk-*` operations) when it is used to divide the combined matrix.
This is transparent to end users but surprising for developers who expect to recover the same sequence of delayed operations that they put in.

```python
import numpy
import dask.array as da
X = da.from_array(numpy.random.rand(100, 20))
Y = da.from_array(numpy.random.rand(100, 20))
combined = numpy.concatenate((X, Y), axis=1)
all_sums = combined.sum(axis=0).compute()
normalized = combined / all_sums

for x in normalized.dask.keys():
    print(x)
## ('truediv-78719db305246fd474d0136dd804a738', 0, 0)
## ('truediv-78719db305246fd474d0136dd804a738', 0, 1)
## ('concatenate-fea8d498137cb283a1ccd462d1f978aa', 0, 0)
## ('concatenate-fea8d498137cb283a1ccd462d1f978aa', 0, 1)
## ('array-eb5604e3cb2e184e5a2e5b6769b87871', 0, 0)
## ('array-81bcf7d160e27bf3266f48572db16a6e', 0, 0)
## ('array-1f6097f7707042f07edf65a6a9eb020e', 0)
## ('rechunk-merge-22938068fb344dacd51a37e23fefd574', 0)
## ('rechunk-merge-22938068fb344dacd51a37e23fefd574', 1)
## ('rechunk-split-22938068fb344dacd51a37e23fefd574', 0)
## ('rechunk-split-22938068fb344dacd51a37e23fefd574', 1)
```

These experiences were the motivation for the **DelayedArray** package, which captures delayed operations in a much simpler format for robust BiocPy package development.
And hey, at least we control the representation, so we can coordinate changes across the BiocPy suite of packages.
Note that we still use **dask** to do some Python-level compute (e.g., sums, variances) via the NumPy interface;
this should be fine as it's more stable than the internals.
