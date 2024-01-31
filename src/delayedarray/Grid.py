from typing import Tuple, Sequence, Optional, List, Generator
import bisect
import math


class Grid:
    """
    Virtual base class for array grids. Each grid subdivides an array to
    determine how it should be iterated over; this is useful for ensuring that
    iteration respects the physical layout of an array.  Check out
    :py:class:`~SimpleGrid` and :py:class:`~CompositeGrid` for subclasses.
    """
    pass


class SimpleGrid(Grid): 
    """
    A simple grid to subdivide an array, involving arbitrary boundaries on each
    dimension. Each grid element is defined by boundaries on each dimension.
    """

    def __init__(self, boundaries: Tuple[Sequence[int], ...]):
        """
        Args:
            boundaries: 
                Tuple of length equal to the number of dimensions. Each entry
                should be a strictly increasing sequence of integers specifying
                the position of the grid boundaries; the last element should
                be equal to the extent of the dimension for the array.
        """
        shape = []
        maxgap = []
        for i, curs in enumerate(boundaries):
            shape.append(curs[-1])
            last = 0
            curmax = 0
            for d in curs:
                gap = d - last
                if gap > curmax:
                    curmax = gap
                last = d
            maxgap.append(curmax)

        self._shape = (*shape,)
        self._boundaries = boundaries
        self._maxgap = (*maxgap,)


    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Shape of the grid, equivalent to the array's shape.
        """
        return self._shape


    @property
    def boundaries(self) -> Tuple[Sequence[int], ...]:
        """
        Returns:
            Boundaries on each dimension of the grid.
        """
        return self._boundaries


    def subset(self, subset: Tuple[Optional[Sequence[int]], ...]) -> "SimpleGrid":
        """
        Subset a grid to reflect the same operation on the associated array.
        For any given dimension, consecutive elements in the subset are only
        placed in the same grid interval in the subsetted grid if they belong
        to the same grid interval in the original grid.

        Args:
            subset:
                Tuple of length equal to the number of grid dimensions. Each
                entry should be a (possibly unsorted) sequence of integers,
                specifying the subset to apply to each dimension of the grid.
                Alternatively, an entry may be None if no subsetting is to be
                applied to the corresponding dimension.

        Returns:
            A new ``SimpleGrid`` object.
        """
        new_boundaries = []
        if len(subset) != len(self._shape):
            raise ValueError("'shape' and 'subset' should have the same length")

        for i, bounds in enumerate(self._boundaries):
            cursub = subset[i]
            if cursub is None:
                new_boundaries.append(bounds)
                continue

            my_boundaries = []
            counter = 0
            last_chunk = -1
            for y in cursub:
                cur_chunk = bisect.bisect_right(bounds, y)
                if cur_chunk != last_chunk:
                    if counter > 0:
                        my_boundaries.append(counter)
                    last_chunk = cur_chunk
                counter += 1 
            my_boundaries.append(counter)

            new_boundaries.append(my_boundaries)

        return SimpleGrid((*new_boundaries,))


    def _recursive_iterate(self, dimension: int, used: List[bool], starts: List[int], ends: List[int], buffer_elements: int):
        bounds = self._boundaries[dimension]
        full_end = self._shape[dimension]

        if used:
            # We assume the worst case and consider the space available if all
            # the remaining dimensions use their maximum gap. Note that this
            # only differs from buffer_elements when dimension > 0.
            conservative_buffer_elements = buffer_elements
            for d in range(dimension):
                conservative_buffer_elements /= self._maxgap[d]
            conservative_buffer_elements = max(1, conservative_buffer_elements)

            start = 0
            pos = 0
            nb = len(bounds)

            while True:
                if pos == nb:
                    # Wrapping up the last block, if the grid-breaking code
                    # has not already iterated to the end of the dimension.
                    if start != full_end:
                        starts[dimension] = start
                        ends[dimension] = full_end
                        if dimension == 0:
                            yield (*starts,), (*ends,)
                        else:
                            yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (full_end - start))
                    break

                # Check if we can keep going to make a larger block.
                current_end = bounds[pos]
                if current_end - start <= conservative_buffer_elements:
                    pos += 1
                    continue

                previous_end = bounds[pos - 1]

                # Break grid intervals to force compliance with the buffer element limit.
                if previous_end == start:
                    while start < current_end:
                        starts[dimension] = start
                        breaking_end = min(current_end, start + conservative_buffer_elements)
                        ends[dimension] = breaking_end
                        if dimension == 0:
                            yield (*starts,), (*ends,)
                        else:
                            # Next level of recursion uses buffer_elements, not its 
                            # conservative counterpart, as the next level actually has 
                            # knowledge of the boundaries for that dimension.
                            yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (breaking_end - start))
                        start = breaking_end
                    pos += 1
                    continue

                # Falling back to the last boundary that fit in the buffer limit.
                starts[dimension] = start
                ends[dimension] = previous_end
                if dimension == 0:
                    yield (*starts,), (*ends,)
                else:
                    yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (previous_end - start))
                start = previous_end

        else:
            # If this dimension is not used, we return its entire extent.
            starts[dimension] = 0
            ends[dimension] = full_end
            if dimension == 0:
                yield (*starts,), (*ends,)
            else:
                yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // full_end)


    def iterate(self, dimensions: Tuple[int, ...], buffer_elements: int = 1e6) -> Generator[Tuple, None, None]:
        """
        Iterate over an array grid. This assembles blocks of contiguous grid
        intervals to reduce the number of iterations (and associated overhead)
        at the cost of increased memory usage during data extraction.

        Args:
            dimensions:
                Dimensions over which to perform the iteration. Any dimensions
                not listed here are extracted in their entirety, i.e., each
                block consists of the full extent of unlisted dimensions.

            buffer_elements:
                Total number of elements in each block. Larger values increase
                the block size and reduce the number of iterations, at the cost
                of increased memory usage at each iteration.

        Returns:
            A generator that returns a tuple of length equal to the number of
            dimensions. Each element contains the start and end of the block
            on its corresponding dimension.
        """
        ndim = len(self._shape)
        used = [False] * ndim
        for i in dimensions:
            used[i] = True

        for i, d in enumerate(self._shape):
            if not used[i]:
                buffer_elements //= d

        starts = [0] * ndim
        ends = [0] * ndim
        yield from self._recursive_iterate(self, dimension=ndim - 1, used=used, starts=starts, ends=ends, buffer_elements=buffer_elements)
