import bisect

class Grid:
    pass


class SimpleGrid(Grid): 
    def __init__(self, shape: Tuple[int, ...], spacing: Tuple[Sequence[int], ...], check: bool = True):
        if len(shape) != len(spacing):
            raise ValueError("'shape' and 'spacing' should have the same length")

        for i, d in enumerate(shape):
            if d:
                if spacing[i][0] != 0:
                    raise ValueError("first element of each 'spacing' should be zero")
            else:
                if len(spacing[i]) != 0:
                    raise ValueError("'spacing' should have zero length of any zero-valued entry of 'shape'")

        self._shape = shape
        self._spacing = spacing


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

    
    def iterate(self, dimensions: Union[int, Tuple[int, ...]], buffer_elements: int) -> Generator[Tuple]:













class CompositeGrid(Grid):
    def __init__(components: List[Grid], along: int):
        self._grids = grids
        self._along = along

