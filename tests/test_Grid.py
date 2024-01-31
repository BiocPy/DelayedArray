import delayedarray
import random
import pytest
import numpy


def test_SimpleGrid_basic():
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)))
    assert grid.shape == (50, 20)
    assert len(grid.boundaries[0]) == 5
    assert len(grid.boundaries[1]) == 7


def test_SimpleGrid_subset():
    # No-op subsetting.
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)))
    subgrid = grid.subset((None, None))
    assert subgrid.boundaries == grid.boundaries

    def full_map(boundaries):
        assignments = []
        chunk = 0
        pos = 0
        for b in boundaries:
            while pos < b:
                assignments.append(chunk)
                pos += 1
            chunk += 1
        return assignments

    def assert_valid_reassignments(subset, original, updated):
        original_assignments = full_map(original)
        updated_assignments = full_map(updated)
        last_ochunk = original_assignments[subset[0]]
        last_uchunk = updated_assignments[0]
        for i in range(1, len(subset)):
            cur_ochunk = original_assignments[subset[i]]
            cur_uchunk = updated_assignments[i] 
            assert (cur_ochunk == last_ochunk) == (cur_uchunk == last_uchunk)
            last_ochunk = cur_ochunk
            last_uchunk = cur_uchunk

    # Consecutive subsetting.
    sub = (range(0, 50, 3), range(0, 20, 5))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])

    # Scrambled subsetting (full)
    sub = (random.sample(range(50), 50), random.sample(range(20), 20))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])

    # Scrambled subsetting (partial)
    sub = (random.sample(range(50), 20), random.sample(range(20), 10))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_SimpleGrid_iterate_2d(buffer_elements):
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)))

    # Full iteration on both dimensions.
    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(0,1), buffer_elements=buffer_elements):
        full_size = 1
        for s, e in block:
            full_size *= e - s
        assert full_size <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()

    # Iteration on only one dimension.
    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(0,), buffer_elements=buffer_elements):
        assert block[1] == (0, grid.shape[1]) 
        gap = block[0][1] - block[0][0]
        assert gap == 1 or gap * grid.shape[1] <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()

    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(1,), buffer_elements=buffer_elements):
        assert block[0] == (0, grid.shape[0]) 
        gap = block[1][1] - block[1][0]
        assert gap == 1 or gap * grid.shape[0] <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_SimpleGrid_iterate_3d(buffer_elements):
    grid = delayedarray.SimpleGrid((range(1, 11, 1), range(10, 51, 10), range(2, 21, 3)))

    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(0,1,2), buffer_elements=buffer_elements):
        full_size = 1
        for s, e in block:
            full_size *= e - s
        assert full_size <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()

    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(0,2), buffer_elements=buffer_elements):
        assert block[1] == (0, grid.shape[1]) 
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()

    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(1,2), buffer_elements=buffer_elements):
        assert block[0] == (0, grid.shape[0]) 
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()

    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(0,1), buffer_elements=buffer_elements):
        assert block[2] == (0, grid.shape[2]) 
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()