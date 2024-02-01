import delayedarray


def test_RegularTicks():
    out = delayedarray.RegularTicks(7, 50)
    ref = list(range(7, 50, 7)) + [50]
    assert len(out) == len(ref)
    assert list(out) == ref
    assert out[-1] == 50 # check reverse indexing.

    out = delayedarray.RegularTicks(10, 50)
    ref = list(range(10, 51, 10))
    assert list(out) == ref
    assert len(out) == len(ref)

    out = delayedarray.RegularTicks(20, 50)
    ref = [20, 40, 50]
    assert list(out) == ref
    assert len(out) == len(ref)

    out = delayedarray.RegularTicks(25, 50)
    ref = [25, 50]
    assert list(out) == ref
    assert len(out) == len(ref)

    out = delayedarray.RegularTicks(100, 50)
    ref = [50]
    assert list(out) == ref
    assert len(out) == len(ref)

    out = delayedarray.RegularTicks(1, 0)
    ref = []
    assert list(out) == ref
    assert len(out) == len(ref)
