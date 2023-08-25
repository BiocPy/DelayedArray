# Tutorial

Oh.
There's not much to say that wasn't already covered in the [Overview](readme.md), but I guess I could talk about it a bit more.
Let's start off with a count matrix:

```python
import numpy
counts = (10 * numpy.random.rand(100, 10)).astype(numpy.int32)
```

And then we wrap it in a `DelayedArray`:

```python
import delayedarray
wrapped = delayedarray.DelayedArray(counts)
type(wrapped)
## <class 'delayedarray.DelayedArray.DelayedArray'>
```

We can now apply some transformations.
In genomics, a typical quality control task is to slice the matrix to remove uninteresting features (rows) or samples (columns):

```python
filtered = wrapped[1:100:2,1:8]
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
```

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

If we need NumPy methods that are not exposed by the `DelayedArray` interface, we can just convert our `DelayedArray`s to regular NumPy arrays:

```python
realized = numpy.array(transformed)
type(realized)
## <class 'numpy.ndarray'>
```

Or even better, we can convert them to **dask** arrays, which preserves the delayed nature of the operations to avoid unnecessary copies/evalution.
(You might wonder why we didn't just do this in the first place - check out the [developer notes](developers.md) for commentary.)

```python
daskified = transformed.as_dask_array()
type(daskified)
## <class 'dask.array.core.Array'>
```
