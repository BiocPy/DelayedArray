import delayedarray


def test_default_buffer_size():
    assert delayedarray.default_buffer_size() == 1e8

    old = delayedarray.default_buffer_size(500)
    assert old == 1e8
    assert delayedarray.default_buffer_size() == 500

    delayedarray.default_buffer_size(old)
    assert delayedarray.default_buffer_size() == old
