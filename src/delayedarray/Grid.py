from typing import Tuple, Sequence, Optional, List, Generator, Dict
import bisect
import abc

from ._subset import _is_single_subset_noop


class AbstractGrid(abc.ABC):
    """
    Abstract base class for array grids. Each grid subdivides an array to
    determine how it should be iterated over; this is useful for ensuring that
    iteration respects the physical layout of an array. 

    Subclasses should define the ``shape``, ``boundaries`` and ``cost``
    properties, as well as the ``subset``, ``transpose`` and ``iterate``
    methods; see the :py:class:`~SimpleGrid` and :py:class:`~CompositeGrid`
    subclasses for examples.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass


    @property
    @abc.abstractmethod
    def cost(self) -> int:
        pass


    @property
    @abc.abstractmethod
    def boundaries(self) -> Tuple[Sequence[int], ...]:
        pass


    @abc.abstractmethod
    def transpose(self, perm: Tuple[int, ...]) -> "AbstractGrid":
        pass


    @abc.abstractmethod
    def subset(self, subset: Tuple[Sequence[int], ...]) -> "AbstractGrid":
        pass


    @abc.abstractmethod
    def iterate(self, dimensions: Tuple[int, ...], buffer_elements: int = 1e6) -> Generator[Tuple, None, None]:
        pass


############################################################
############################################################


class SimpleGrid(AbstractGrid): 
    """
    A simple grid to subdivide an array, involving arbitrary boundaries on each
    dimension. Each grid element is defined by boundaries on each dimension.
    """

    def __init__(self, boundaries: Tuple[Sequence[int], ...], cost_factor: float, internals: Optional[Dict] = None):
        """
        Args:
            boundaries: 
                Tuple of length equal to the number of dimensions. Each entry
                should be a strictly increasing sequence of integers specifying
                the position of the grid boundaries; the last element should be
                equal to the extent of the dimension for the array. A tuple
                entry may also be an empty list for a zero-extent dimension.

            cost_factor:
                Positive number representing the cost of iteration over each
                element of the grid's array. The actual cost is defined by the
                product of the cost factor by the array size. This is used to
                choose between iteration schemes; as a reference, extraction
                from an in-memory NumPy array has a cost factor of 1.

            internals:
                Internal use only.
        """
        self._boundaries = boundaries

        if internals is not None and "shape" in internals:
            shape = internals["shape"]
        else:
            new_shape = []
            for bounds in boundaries:
                if len(bounds):
                    new_shape.append(bounds[-1])
                else:
                    new_shape.append(0)
            shape = (*new_shape,)
        self._shape = shape

        if internals is not None and "maxgap" in internals:
            maxgap = internals["maxgap"]
        else:
            maxgap = (*(SimpleGrid._define_maxgap(b) for b in boundaries),)
        self._maxgap = maxgap

        if internals is not None and "cost" in internals:
            cost = internals["cost"]
        else:
            cost = cost_factor
            for s in shape:
                cost *= s
        self._cost = cost
        self._cost_factor = cost_factor


    @staticmethod
    def _define_maxgap(bounds):
        last = 0
        curmax = 0
        for d in bounds:
            gap = d - last
            if gap > curmax:
                curmax = gap
            last = d
        return curmax


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


    def transpose(self, perm: Tuple[int, ...]) -> "SimpleGrid":
        """
        Transpose a grid to reflect the same operation on the associated array.

        Args:
            perm:
                Tuple of length equal to the dimensionality of the array,
                containing the permutation of dimensions.

        Returns:
            A new ``SimpleGrid`` object.
        """
        return SimpleGrid(
            (*(self._boundaries[p] for p in perm),), 
            self._cost_factor,
            internals = { # Save ourselves the trouble of recomputing these.
                "shape": (*(self._shape[p] for p in perm),),
                "maxgap": (*(self._maxgap[p] for p in perm),),
                "cost": self._cost,
            }
        )


    def subset(self, subset: Tuple[Sequence[int], ...]) -> "SimpleGrid":
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

        Returns:
            A new ``SimpleGrid`` object.
        """
        if len(subset) != len(self._shape):
            raise ValueError("'shape' and 'subset' should have the same length")

        new_boundaries = []
        new_shape = []
        new_maxgap = []
        for i, bounds in enumerate(self._boundaries):
            cursub = subset[i]
            if _is_single_subset_noop(self._shape[i], cursub):
                new_boundaries.append(bounds)
                new_shape.append(self._shape[i])
                new_maxgap.append(self._maxgap[i])
            else: 
                my_boundaries = []
                counter = 0
                if len(bounds):
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
                new_shape.append(counter)
                new_maxgap.append(SimpleGrid._define_maxgap(my_boundaries))

        return SimpleGrid(
            (*new_boundaries,), 
            self._cost_factor,
            internals = {
                "shape": (*new_shape,),
                "maxgap": (*new_maxgap,),
            }
        )


    def _recursive_iterate(self, dimension: int, used: List[bool], starts: List[int], ends: List[int], buffer_elements: int, prescale: List[int]):
        bounds = self._boundaries[dimension]
        full_end = self._shape[dimension]

        if used[dimension]:
            # 'prescale' holds the worst-case minimum block sizes of the
            # remaining dimensions (i.e., in the subsequent recursion levels).
            # These are the worst-case because we assume that all remaining
            # dimensions are using their maximum gap for a single grid interval
            # to form a block, hence it's a conservative estimate of the space
            # we have left to play with on the current dimenion. Note that this
            # should only differ from buffer_elements when dimension > 0.
            conservative_buffer_elements = max(1, buffer_elements // prescale[dimension])

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
                            yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (full_end - start), prescale)
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
                            yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (breaking_end - start), prescale)
                        start = breaking_end
                    pos += 1
                    continue

                # Falling back to the last boundary that fit in the buffer limit.
                starts[dimension] = start
                ends[dimension] = previous_end
                if dimension == 0:
                    yield (*zip(starts, ends),)
                else:
                    yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // (previous_end - start), prescale)
                start = previous_end

        else:
            # If this dimension is not used, we return its entire extent.
            starts[dimension] = 0
            ends[dimension] = full_end
            if dimension == 0:
                yield (*zip(starts, ends),)
            else:
                yield from self._recursive_iterate(dimension - 1, used, starts, ends, buffer_elements // full_end, prescale)


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
        if 0 in self._shape:
            return

        ndim = len(self._shape)
        used = [False] * ndim
        for d in dimensions:
            used[d] = True

        # See comments above about 'prescale'. For each dimension, this
        # contains the worst-case minimum block size from the remaining
        # dimensions. For used dimensions, this is computed from the max gap;
        # for unused dimensions, the full extent is applied.
        prescale = [1]
        for i in range(ndim - 1):
            last = prescale[-1]
            if used[i]:
                last *= self._maxgap[i]
            else:
                last *= self._shape[i]
            prescale.append(last)

        starts = [0] * ndim
        ends = [0] * ndim
        yield from self._recursive_iterate(ndim - 1, used, starts, ends, buffer_elements, prescale)


############################################################
############################################################


class CompositeGrid(AbstractGrid): 
    """
    A grid to subdivide an array, constructed by combining component grids
    along a specified dimension. This aims to mirror the same combining
    operation for the arrays associated with the component grids.
    """

    def __init__(self, components: Tuple[AbstractGrid, ...], along: int, internals: Optional[Dict] = None):
        """
        Args:
            components: 
                Component grids to be combined to form the composite grid.
                Each grid should have the same dimension extents, except for
                the ``along`` dimension.

            along:
                Dimension over which to combine entries of ``components``.

            internals:
                Internal use only.
        """
        self._components = components
        self._along = along

        if internals is not None and "shape" in internals:
            shape = internals["shape"]
        else:
            first = components[0]
            new_shape = list(first.shape)
            for i in range(1, len(components)):
                current = components[i]
                for j, d in enumerate(current.shape):
                    if j == along:
                        new_shape[j] += d
                    elif new_shape[j] != d:
                        raise ValueError("entries of 'components' should have the same shape on all dimensions except 'along'")
            shape = (*new_shape,)
        self._shape = shape

        if internals is not None and "boundaries" in internals:
            boundaries = internals["boundaries"]
        else:
            boundaries = None
        self._boundaries = boundaries

        if internals is not None and "cost" in internals:
            cost = internals["cost"]
        else:
            cost = 0
            for comp in self._components:
                cost += comp.cost
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
                offset += comp.shape[self._along]

            new_boundaries[self._along] = replacement
            self._boundaries = (*new_boundaries,)

        return self._boundaries


    @property
    def cost(self) -> float:
        """
        Returns:
            Cost of iteration over the underlying array. This is defined
            as the sum of the costs of the component arrays.
        """
        return self._cost


    def _maxcost(self) -> Tuple[int, float]:
        chosen = 0 
        maxcost = 0
        for i, comp in enumerate(self._components):
            if isinstance(comp, CompositeGrid):
                tmp, curcost = comp._maxcost()
            else:
                curcost = comp.cost
            if curcost > maxcost:
                maxcost = curcost
                chosen = i
        return chosen, maxcost


    def transpose(self, perm: Tuple[int, ...]) -> "CompositeGrid":
        """
        Transpose a grid to reflect the same operation on the associated array.

        Args:
            perm:
                Tuple of length equal to the dimensionality of the array,
                containing the permutation of dimensions.

        Returns:
            A new ``CompositeGrid`` object.
        """
        new_components = [grid.transpose(perm) for grid in self._components]

        new_along = 0
        for i, p in enumerate(perm):
            if p == self._along:
                new_along = i

        internals = { "cost": self._cost }
        if self._boundaries is not None:
            internals["boundaries"] = (*(self._boundaries[p] for p in perm),)

        return CompositeGrid(
            new_components,
            new_along,
            internals = internals,
        )


    def subset(self, subset: Tuple[Sequence[int], ...]) -> "CompositeGrid":
        """
        Subset a grid to reflect the same operation on the associated array.
        This splits up the subset sequence for the ``along`` dimension and
        distributes it to each of the component grids.

        Args:
            subset:
                Tuple of length equal to the number of grid dimensions. Each
                entry should be a (possibly unsorted) sequence of integers,
                specifying the subset to apply to each dimension of the grid.

        Returns:
            A new ``CompositeGrid`` object.
        """
        if len(subset) != len(self._shape):
            raise ValueError("'shape' and 'subset' should have the same length")

        if _is_single_subset_noop(self._shape[self._along], subset[self._along]):
            new_components = []
            new_subset = list(subset)
            for grid in self._components:
                new_subset[self._along] = range(grid.shape[self._along])
                new_components.append(grid.subset((*new_subset,)))
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

        new_shape = []
        for i, s in enumerate(subset):
            if s is None:
                new_shape.append(self._shape[i])
            else:
                new_shape.append(len(s))

        return CompositeGrid(
            new_components, 
            self._along, 
            internals = {
                "shape": (*new_shape,),
            }
        )


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
        if 0 in self._shape:
            return

        if self._along in dimensions:
            first = True
            offset = 0
            for grid in self._components:
                for block in grid.iterate(dimensions=dimensions, buffer_elements=buffer_elements):
                    if offset == 0:
                        yield block 
                    else:
                        copy = list(block)
                        comp = copy[self._along]
                        copy[self._along] = (comp[0] + offset, comp[1] + offset)
                        yield (*copy,)
                offset += grid.shape[self._along]
            return
        
        temp = SimpleGrid(self.boundaries, cost_factor=1)
        yield from temp.iterate(dimensions=dimensions, buffer_elements=buffer_elements)
