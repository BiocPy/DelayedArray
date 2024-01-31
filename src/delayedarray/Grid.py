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
        buffer_elements = max(1, buffer_elements)

        if used:
            # We assume the worst case and consider the space available if all
            # the remaining dimensions use their maximum gap. Note that this
            # only differs from buffer_elements when dimension > 0.
            conservative_buffer_elements = buffer_elements
            for d in range(dimension):
                conservative_buffer_elements /= self._maxgap[d]

            start = 0
            ns = len(curs)
            for pos in range(1, n):
                if curs[pos] - start <= conservative_buffer_elements: # i.e., we can keep going to make a larger block.
                    if pos + 1 < ns: # i.e., it's not the last element, in which case we would be forced to yield.
                        continue

                end = curs[pos - 1]
                if end == start:
                    # Break chunks to force compliance with the buffer element limit.
                    full_end = curs[pos]
                    while start < full_end:
                        starts[dimension] = start
                        end = min(full_end, start + conservative_buffer_elements)
                        ends[dimension] = end
                        if dimension == 0:
                            yield starts, ends
                        else:
                            # Next level of recursion still uses buffer_elements, 
                            # not its conservative counterpart, as the next level 
                            # actually has access to the spacings for that dimension.
                            yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (end - start))
                        start = end
                    continue

                # Falling back to the last breakpoint that fit in the buffer limit.
                starts[dimension] = start
                ends[dimension] = end
                if dimension == 0:
                    yield starts, ends
                else:
                    yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // (end - start))
                start = end

        else:
            # If it's not used, we return the entire extent.
            full = self._shape[0]
            starts[0] = 0
            ends[0] = full
            if dimension == 0:
                yield starts, ends
            else:
                yield from self._recursive_iterate(dimension - 1, starts, ends, buffer_elements // full)


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
        yield from self._recursive_iterate(self, dimension=ndim - 1, used=used, starts=starts, ends=ends, buffer_elements=buffer_elements)
