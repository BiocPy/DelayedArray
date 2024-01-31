from typing import Tuple, Sequence, Optional, List, Generator
import bisect
import math


class Grid:
    """
    Virtual base class for array grids. Each grid subdivides an array to
    determine how it should be iterated over; this is useful for ensuring that
    iteration respects the physical layout of an array. 

    Subclasses are expected to define the ``shape``, ``boundaries`` and
    ``cost`` properties; check out :py:class:`~SimpleGrid` and
    :py:class:`~CompositeGrid` for examples.
    """
    pass


class SimpleGrid(Grid): 
    """
    A simple grid to subdivide an array, involving arbitrary boundaries on each
    dimension. Each grid element is defined by boundaries on each dimension.
    """

    def __init__(self, boundaries: Tuple[Sequence[int], ...], cost_factor: float):
        """
        Args:
            boundaries: 
                Tuple of length equal to the number of dimensions. Each entry
                should be a strictly increasing sequence of integers specifying
                the position of the grid boundaries; the last element should
                be equal to the extent of the dimension for the array.

            cost_factor:
                Positive number representing the cost of iteration over each
                element of the grid's array. The actual cost is defined by the
                product of the cost factor by the array size. This is used to
                choose between iteration schemes.
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

        cost = cost_factor
        for s in shape:
            cost *= s
        self._cost = cost


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


    @property
    def cost(self) -> float:
        """
        Returns:
            Cost of iteration over the underlying array.
        """
        return self._cost


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

        if used[dimension]:
            # We assume the worst case and consider the space available if all
            # the remaining dimensions use their maximum gap. Note that this
            # only differs from buffer_elements when dimension > 0.
            conservative_buffer_elements = buffer_elements
            for d in range(dimension):
                if used[d]:
                    denom = self._maxgap[d]
                else:
                    denom = self._shape[d]
                conservative_buffer_elements //= denom
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
                            yield (*zip(starts, ends),)
                        else:
                            yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (full_end - start))
                    break

                # Check if we can keep going to make a larger block.
                current_end = bounds[pos]
                if current_end - start <= conservative_buffer_elements:
                    pos += 1
                    continue

                if pos:
                    previous_end = bounds[pos - 1]
                else:
                    previous_end = 0

                # Break grid intervals to force compliance with the buffer element limit.
                if previous_end == start:
                    while start < current_end:
                        starts[dimension] = start
                        breaking_end = min(current_end, start + conservative_buffer_elements)
                        ends[dimension] = breaking_end
                        if dimension == 0:
                            yield (*zip(starts, ends),)
                        else:
                            # Next level of recursion uses buffer_elements, not its 
                            # conservative counterpart, as the next level actually has 
                            # knowledge of the boundaries for that dimension.
                            yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (breaking_end - start))
                        start = breaking_end
                    pos += 1
                    continue

                # Falling back to the last boundary that fit in the buffer limit.
                starts[dimension] = start
                ends[dimension] = previous_end
                if dimension == 0:
                    yield (*zip(starts, ends),)
                else:
                    yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (previous_end - start))
                start = previous_end

        else:
            # If this dimension is not used, we return its entire extent.
            starts[dimension] = 0
            ends[dimension] = full_end
            if dimension == 0:
                yield (*zip(starts, ends),)
            else:
                yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // full_end)


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

        starts = [0] * ndim
        ends = [0] * ndim
        yield from self._recursive_iterate(ndim - 1, used, starts, ends, buffer_elements)


class CompositeGrid(Grid): 
    def __init__(self, components: Tuple[Grid, ...], along: int):
        first = components[0]
        shape = list(first.shape)
        for i in range(1, len(components)):
            current = components[i]
            for j, d in enumerate(current.shape):
                if j == along:
                    shape[j] += d
                elif shape[j] != d:
                    raise ValueError("entries of 'components' should have the same shape on all dimensions except 'along'")

        self._shape = (*shape,)
        self._components = components
        self._along = along
        self._boundaries = None
        self._cost = None


    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Shape of the grid, equivalent to the array's shape.
        """
        return self._shape


    def boundaries(self) -> Tuple[Sequence[int], ...]:
        """
        Returns:
            Boundaries on each dimension of the grid. For the ``along``
            dimension, this is a concatenation of the boundaries for the
            component grids. For all other dimensions, the boundaries are
            set to those of the most costly component grid.
        """
        if self._boundaries is None: # Lazy evaluation
            chosen, maxcost = self._maxcost()
            new_boundaries = list(self._components[chosen].boundaries)
            replacement = []
            offset = 0

            for i, comp in enumerate(self._components):
                if i == chosen:
                    addition = new_boundaries[self._along]
                else:
                    addition = comp.boundaries[self._along]
                for a in addition:
                    replacement.append(a + offset)
                offset += comp.shape

            new_boundaries[self._along] = replacement
            self._boundaries = (*new_boundaries,)

        return self._boundaries = new_boundaries


    @property
    def cost(self) -> float:
        """
        Returns:
            Cost of iteration over the underlying array. This is defined
            as the sum of the costs of the component arrays.
        """
        if self._cost is None: # lazy evaluation
            cost = 0
            for comp in self._components:
                cost += comp.cost
            self._cost = cost
        return self._cost


    def _max_cost() -> Tuple[int, float]:
        chosen = 0 
        maxcost = 0
        for i, comp in enumerate(self._components):
            if isinstance(comp, CompositeGrid):
                tmp, curcost = comp._max_cost()
            else:
                curcost = comp.cost
            if curcost > maxcost:
                maxcost = curcost
                chosen = i
        return chosen, maxcost


    def subset(self, subset: Tuple[Optional[Sequence[int]], ...]) -> "CompositeGrid":
        """
        Subset a grid to reflect the same operation on the associated array.
        This splits up the subset sequence for the ``along`` dimension and
        distributes it to each of the component grids.

        Args:
            subset:
                Tuple of length equal to the number of grid dimensions. Each
                entry should be a (possibly unsorted) sequence of integers,
                specifying the subset to apply to each dimension of the grid.
                Alternatively, an entry may be None if no subsetting is to be
                applied to the corresponding dimension.

        Returns:
            A new ``CompositeGrid`` object.
        """
        if subset[self._along] is None:
            new_components = [grid.subset(subset) for grid in self._components]
            return CompositeGrid(new_components, self._along)

        component_limits = []
        counter = 0 
        for y in self._components:
            counter += y.shape[self._along]
            component_limits.append(counter)

        last_choice = -1
        new_components = []
        sofar = []
        raw_subset = list(subset)

        for s in subset[self._along]:
            choice = bisect.bisect_left(component_limits, s)
            if choice != last_choice:
                if len(sofar):
                    raw_subset[self._along] = sofar
                    new_components.append(self._components[last_choice].subset((*raw_subset,)))
                    sofar = []
                last_choice = choice
            if choice:
                sofar.append(s - component_limits[choice - 1])
            else:
                sofar.append(s)

        if len(sofar):
            raw_subset[self._along] = sofar
            new_components.append(self._components[last_choice].subset((*raw_subset,)))
        return CompositeGrid(new_components, self._along)


    def iterate(self, dimensions: Tuple[int, ...], buffer_elements: int = 1e6) -> Generator[Tuple, None, None]:
        """
        Iterate over an array grid. This assembles blocks of contiguous grid
        intervals to reduce the number of iterations (and associated overhead)
        at the cost of increased memory usage during data extraction. For any
        iteration over the ``along`` dimension (i.e., ``along`` is in
        ``dimensions``), this function dispatches to the component grids;
        otherwise the iteration is performed based on :py:meth:`~boundaries`.

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
        if self._along in dimensions:
            for grid in self._components:
                yield from grid.iterate(dimensions=dimensions, buffer_elements=buffer_elements)
            return
        
        temp = SimpleGrid(self.boundaries)
        yield from temp.iterate(dimensions=dimensions, buffer_elements=buffer_elements)
