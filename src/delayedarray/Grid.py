import bisect
import math


class Grid:
    pass


class SimpleGrid(Grid): 
    def __init__(self, shape: Tuple[int, ...], spacing: Tuple[Sequence[int], ...], check: bool = True):
        if len(shape) != len(spacing):
            raise ValueError("'shape' and 'spacing' should have the same length")

        maxgap = []
        for i, d in enumerate(shape):
            curs = spacing[i]
            if curs[0] != 0:
                raise ValueError("first element of each 'spacing' should be zero")
            if curs[-1] != d:
                raise ValueError("last element of each 'spacing' should be equal to the corresponding entry of 'shape'")
            curmax = 0
            for j in range(1, len(curs)):
                gap = curs[j] - curs[j-1]
                if gap > curmax:
                    curmax = gap
            maxgap.append(curmax)

        self._shape = shape
        self._spacing = spacing
        self._maxgap = maxgap


    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


    def slice(self, subset: Tuple[Optional[Sequence[int]], ...]) -> SimpleGrid:
        new_spacing = []
        new_shape = []
        if len(subset) != len(self._shape):
            raise ValueError("'shape' and 'subset' should have the same length")

        for i, d in enumerate(self._spacing):
            cursub = subset[i]
            if cursub is None:
                new_spacing.append(d)
                new_shape.append(self._shape[i])
                continue

            my_spacing = []
            counter = 0
            last_chunk = -1
            cur_chunk = -2
            for y in cursub:
                cur_chunk = bisect.bisect_left(cursub, y)
                if cur_chunk != last_chunk:
                    my_spacing.append(counter)
                    last_chunk = cur_chunk
                counter += 1 
            if cur_chunk == last_chunk:
                my_spacing.append(counter)

            new_shape.append(len(cursub))
            new_spacing.append(my_spacing)

        return SimpleGrid(new_shape, new_spacing)


    def _recursive_iterate(self, dimension: int, used: List[bool], starts: List[int], ends: List[int], buffer_elements: int):
        curs = self._spacing[dimension]

        if dimension == 0:
            if used:
                start = 0
                for pos in range(2, len(curs)):
                    # This setup ensures that we always take whole chunks, even
                    # if we technically don't have enough buffer to do so.
                    if curs[pos] - start > buffer_elements:
                        starts[0] = start
                        end = curs[pos - 1]
                        ends[0] = end
                        yield starts, ends
                        start = end

                starts[0] = start
                ends[0] = curs[-1]
                yield starts, ends

            else:
                starts[0] = 0
                ends[0] = self._shape[0]
                yield starts, ends

        else:
            if used:
                # We assume the worst case and consider the space available
                # if all the remaining dimensions use their maximum gap.
                conservative_buffer_elements = buffer_elements
                for d in range(dimension):
                    conservative_buffer_elements /= self._maxgap[d]

                start = 0
                for pos in range(2, len(curs)):
                    # Again, ensuring that we always proceed with whole chunks.
                    if curs[pos] - start > conservative_buffer_elements:
                        starts[dimension] = start
                        end = curs[pos - 1]
                        ends[dimension] = end
                        yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (end - start))
                        start = end

                starts[dimension] = start
                ends[dimension] = curs[-1]
                yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements)

            else:
                starts[0] = 0
                ends[0] = self._shape[dimension]
                yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // self._shape[dimension])


    def iterate(self, dimensions: Union[int, Tuple[int, ...]], buffer_elements: int) -> Generator[Tuple]:
        ndim = len(self._shape)
        used = [False] * ndim
        for i in dimensions:
            used[i] = True

        for i, d in enumerate(self._shape):
            if not used[i]:
                buffer_elements //= d

        starts = [0] * ndim
        ends = [0] * ndim
        yield from self._recursive_iterate(self, dimension, used, starts, ends, buffer_elements)
                        













