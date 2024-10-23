import collections


class RegularTicks(collections.abc.Sequence):
    """
    Regular ticks of equal spacing until a limit is reached, at which point the
    sequence terminates at that limit. This is intended for use as grid
    boundaries in :py:class:`~delayedarray.Grid.SimpleGrid`, where the last
    element of the boundary sequence needs to be equal to the grid extent. (We
    do not use :py:class:`~range` as it may omit the last element if the extent
    is not a multiple of the spacing.)
    """

    def __init__(self, spacing: int, final: int):
        """
        Args:
            spacing: 
                Positive integer specifying the spacing between ticks.

            final:
                Position of the final tick, should be non-negative.
        """
        if spacing <= 0:
            raise ValueError("spacing should be positive")
        if final < 0:
            raise ValueError("final should be positive")
        self._spacing = spacing
        self._final = final 
        self._len = (final // spacing) + (final % spacing > 0)

    def __len__(self) -> int:
        """
        Returns:
            Length of the tick sequence.
        """
        return self._len

    def __getitem__(self, i: int) -> int:
        """
        Args:
            i: Index of the tick of interest.

        Returns:
            Position of tick ``i``.
        """
        if i < 0:
            i += self._len
            if i < 0:
                raise IndexError("'i' is out of range")
        elif i >= self._len:
            raise IndexError("'i' is out of range")
        return min(self._final, self._spacing * (i + 1))

    @property
    def spacing(self) -> int:
        """
        Returns:
            The spacing between ticks.
        """
        return self._spacing

    @property
    def final(self) -> int:
        """
        Returns:
            Position of the final tick.
        """
        return self._final
