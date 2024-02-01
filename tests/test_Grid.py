import delayedarray
import random
import pytest
import numpy


def assert_okay_full_iteration(grid, buffer_elements):
    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    dimensions = (*range(len(grid.shape)),)
    for block in grid.iterate(dimensions=dimensions, buffer_elements=buffer_elements):
        full_size = 1
        for s, e in block:
            full_size *= e - s
        assert full_size <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()


def assert_okay_1of2d_iteration(grid, dimension, buffer_elements):
    other = 1 - dimension
    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    for block in grid.iterate(dimensions=(dimension,), buffer_elements=buffer_elements):
        assert block[other] == (0, grid.shape[other]) 
        gap = block[dimension][1] - block[dimension][0]
        assert gap == 1 or gap * grid.shape[other] <= buffer_elements
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()


def assert_okay_2of3d_iteration(grid, dimension, buffer_elements):
    empty = numpy.zeros(grid.shape, dtype=numpy.int32)
    keep = []
    for i in range(len(grid.shape)):
        if i != dimension:
            keep.append(i)

    for block in grid.iterate(dimensions=(*keep,), buffer_elements=buffer_elements):
        assert block[dimension] == (0, grid.shape[dimension]) 
        sub = (*(slice(s, e) for s, e in block),)
        empty[sub] += 1
    assert (empty == 1).all()


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


############################################################
############################################################


def test_SimpleGrid_basic():
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    assert grid.shape == (50, 20)
    assert len(grid.boundaries[0]) == 5
    assert len(grid.boundaries[1]) == 7
    assert grid.cost == 1000


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_SimpleGrid_iterate_2d(buffer_elements):
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    assert_okay_full_iteration(grid, buffer_elements)
    assert_okay_1of2d_iteration(grid, 0, buffer_elements)
    assert_okay_1of2d_iteration(grid, 1, buffer_elements)


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_SimpleGrid_iterate_3d(buffer_elements):
    grid = delayedarray.SimpleGrid((range(1, 11, 1), range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    assert_okay_full_iteration(grid, buffer_elements)
    assert_okay_2of3d_iteration(grid, 0, buffer_elements)
    assert_okay_2of3d_iteration(grid, 1, buffer_elements)
    assert_okay_2of3d_iteration(grid, 2, buffer_elements)


def test_SimpleGrid_empty():
    grid = delayedarray.SimpleGrid(([], range(2, 21, 3)), cost_factor=1)
    assert grid.shape == (0, 20)
    assert len(grid.boundaries[0]) == 0
    assert len(grid.boundaries[1]) == 7
    assert grid.cost == 0
    assert_okay_full_iteration(grid, buffer_elements=100)


def test_SimpleGrid_transpose():
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)

    # No-op transpose.
    tgrid = grid.transpose((0, 1))
    assert tgrid.shape == grid.shape
    assert tgrid.boundaries == grid.boundaries
    assert tgrid.cost == grid.cost
    assert tgrid._maxgap == delayedarray.SimpleGrid(grid.boundaries, cost_factor=1)._maxgap
    assert_okay_full_iteration(tgrid, buffer_elements=100)

    # Actual transpose.
    tgrid = grid.transpose((1, 0))
    assert tgrid.shape == (grid.shape[1], grid.shape[0])
    assert tgrid.boundaries == (grid.boundaries[1], grid.boundaries[0])
    assert tgrid.cost == grid.cost
    assert tgrid._maxgap == (*reversed(delayedarray.SimpleGrid(grid.boundaries, cost_factor=1)._maxgap),)
    assert_okay_full_iteration(tgrid, buffer_elements=100)


def test_SimpleGrid_subset():
    grid = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)

    # No-op subsetting.
    subgrid = grid.subset((*(range(s) for s in grid.shape),))
    assert subgrid.shape == grid.shape
    assert subgrid.boundaries == grid.boundaries
    assert subgrid.cost == grid.cost
    assert subgrid._maxgap == delayedarray.SimpleGrid(subgrid.boundaries, cost_factor=1)._maxgap
    assert_okay_full_iteration(subgrid, buffer_elements=100)

    # Consecutive subsetting.
    sub = (range(0, 50, 3), range(0, 20, 5))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])
    assert subgrid._maxgap == delayedarray.SimpleGrid(subgrid.boundaries, cost_factor=1)._maxgap
    assert_okay_full_iteration(subgrid, buffer_elements=100)

    # Scrambled subsetting (full)
    sub = (random.sample(range(50), 50), random.sample(range(20), 20))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])
    assert subgrid._maxgap == delayedarray.SimpleGrid(subgrid.boundaries, cost_factor=1)._maxgap
    assert_okay_full_iteration(subgrid, buffer_elements=100)

    # Scrambled subsetting (partial)
    sub = (random.sample(range(50), 20), random.sample(range(20), 10))
    subgrid = grid.subset(sub)
    assert subgrid.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], grid.boundaries[0], subgrid.boundaries[0])
    assert_valid_reassignments(sub[1], grid.boundaries[1], subgrid.boundaries[1])
    assert subgrid._maxgap == delayedarray.SimpleGrid(subgrid.boundaries, cost_factor=1)._maxgap
    assert_okay_full_iteration(subgrid, buffer_elements=100)


############################################################
############################################################


def test_CompositeGrid_basic():
    grid1 = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid((range(2, 21, 6), range(5, 21, 5)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)
    assert combined.shape == (70, 20)
    assert combined.boundaries[0] == [10, 20, 30, 40, 50, 52, 58, 64, 70]
    assert combined.boundaries[1] == grid1.boundaries[1] # more expensive by size.
    assert combined.cost == 1400

    grid2 = delayedarray.SimpleGrid((range(2, 21, 6), range(5, 21, 5)), cost_factor=10)
    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)
    assert combined.boundaries[1] == grid2.boundaries[1]
    assert combined.cost == 5000 

    # Now combining along the other dimension.
    grid3 = delayedarray.SimpleGrid((range(1, 51, 7), range(2, 12, 3)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid1, grid3], along=1)
    assert combined.boundaries[1] == [2, 5, 8, 11, 14, 17, 20, 22, 25, 28, 31]
    assert combined.boundaries[0] == grid1.boundaries[0]


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_CompositeGrid_iterate_2d(buffer_elements):
    grid1 = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid((range(2, 21, 6), range(5, 21, 5)), cost_factor=1)

    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)
    assert_okay_full_iteration(combined, buffer_elements)
    assert_okay_1of2d_iteration(combined, 0, buffer_elements)
    assert_okay_1of2d_iteration(combined, 1, buffer_elements)

    grid3 = delayedarray.SimpleGrid((range(1, 51, 7), range(2, 12, 3)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid3, grid1], along=1)
    assert_okay_full_iteration(combined, buffer_elements)
    assert_okay_1of2d_iteration(combined, 0, buffer_elements)
    assert_okay_1of2d_iteration(combined, 1, buffer_elements)


@pytest.mark.parametrize("buffer_elements", [5, 10, 50, 100, 500])
def test_CompositeGrid_iterate_3d(buffer_elements):
    grid1 = delayedarray.SimpleGrid((range(1, 11, 1), range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid((range(1, 11, 3), range(6, 51, 11), range(4, 21, 4)), cost_factor=1)
    grid3 = delayedarray.SimpleGrid((range(2, 11, 2), range(1, 51, 7), range(5, 21, 5)), cost_factor=1)

    combined = delayedarray.CompositeGrid([grid1, grid2, grid3], along=0)
    assert_okay_full_iteration(combined, buffer_elements)
    assert_okay_2of3d_iteration(combined, 0, buffer_elements)
    assert_okay_2of3d_iteration(combined, 1, buffer_elements)
    assert_okay_2of3d_iteration(combined, 2, buffer_elements)

    combined = delayedarray.CompositeGrid([grid1, grid2, grid3], along=1)
    assert_okay_full_iteration(combined, buffer_elements)
    assert_okay_2of3d_iteration(combined, 0, buffer_elements)
    assert_okay_2of3d_iteration(combined, 1, buffer_elements)
    assert_okay_2of3d_iteration(combined, 2, buffer_elements)

    combined = delayedarray.CompositeGrid([grid1, grid2, grid3], along=2)
    assert_okay_full_iteration(combined, buffer_elements)
    assert_okay_2of3d_iteration(combined, 0, buffer_elements)
    assert_okay_2of3d_iteration(combined, 1, buffer_elements)
    assert_okay_2of3d_iteration(combined, 2, buffer_elements)


def test_CompositeGrid_transpose():
    grid1 = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid((range(2, 21, 6), range(5, 21, 5)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)

    tcombined = combined.transpose((0, 1))
    assert tcombined._along == 0
    assert tcombined.shape == combined.shape
    assert tcombined.boundaries == combined.boundaries
    assert tcombined.cost == combined.cost
    assert_okay_full_iteration(tcombined, buffer_elements=100)

    tcombined = combined.transpose((1, 0))
    assert tcombined.shape == (combined.shape[1], combined.shape[0])
    assert tcombined._along == 1
    assert tcombined.boundaries == (combined.boundaries[1], combined.boundaries[0])
    assert tcombined.cost == combined.cost
    assert_okay_full_iteration(tcombined, buffer_elements=100)

    grid3 = delayedarray.SimpleGrid((range(1, 51, 7), range(2, 12, 3)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid3, grid1], along=1)
    tcombined = combined.transpose((1, 0))
    assert tcombined.shape == (combined.shape[1], combined.shape[0])
    assert tcombined._along == 0
    assert tcombined.boundaries == (combined.boundaries[1], combined.boundaries[0])
    assert tcombined.cost == combined.cost
    assert_okay_full_iteration(tcombined, buffer_elements=100)


def test_CompositeGrid_subset():
    grid1 = delayedarray.SimpleGrid((range(10, 51, 10), range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid((range(2, 21, 6), range(5, 21, 5)), cost_factor=1)
    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)

    # No-op subsetting.
    subcombined = combined.subset((*(range(s) for s in combined.shape),))
    assert combined.shape == subcombined.shape
    assert subcombined.boundaries == combined.boundaries
    assert subcombined.cost == combined.cost
    assert_okay_full_iteration(subcombined, buffer_elements=100)

    # Consecutive subsetting.
    sub = (range(0, 50, 3), range(0, 20, 5))
    subcombined = combined.subset(sub)
    assert subcombined.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], combined.boundaries[0], subcombined.boundaries[0])
    assert_valid_reassignments(sub[1], combined.boundaries[1], subcombined.boundaries[1])
    assert_okay_full_iteration(subcombined, buffer_elements=100)

    # Scrambled subsetting (full)
    sub = (random.sample(range(50), 50), random.sample(range(20), 20))
    subcombined = combined.subset(sub)
    assert subcombined.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], combined.boundaries[0], subcombined.boundaries[0])
    assert_valid_reassignments(sub[1], combined.boundaries[1], subcombined.boundaries[1])
    assert_okay_full_iteration(subcombined, buffer_elements=100)

    # Scrambled subsetting (partial)
    sub = (random.sample(range(50), 20), random.sample(range(20), 10))
    subcombined = combined.subset(sub)
    assert subcombined.shape == (len(sub[0]), len(sub[1]))
    assert_valid_reassignments(sub[0], combined.boundaries[0], subcombined.boundaries[0])
    assert_valid_reassignments(sub[1], combined.boundaries[1], subcombined.boundaries[1])
    assert_okay_full_iteration(subcombined, buffer_elements=100)


def test_CompositeGrid_empty():
    grid1 = delayedarray.SimpleGrid(([], range(2, 21, 3)), cost_factor=1)
    grid2 = delayedarray.SimpleGrid(([], range(6, 21, 7)), cost_factor=1)

    combined = delayedarray.CompositeGrid([grid1, grid2], along=0)
    assert combined.shape == (0, 20)
    assert combined.cost == 0
    assert_okay_full_iteration(combined, buffer_elements=100)

    combined = delayedarray.CompositeGrid([grid1, grid2], along=1)
    assert combined.shape == (0, 40)
    assert combined.cost == 0
    assert_okay_full_iteration(combined, buffer_elements=100)
