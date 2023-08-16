from collections import Sequence

class UnaryIsometricAddition:
    """Unary isometric addition, where a scalar or 1-dimensional array is added to a seed array.

    Attributes:
        seed: 
            An array-like object.

        value: 
            A scalar or 1-dimensional array to be added to the ``seed``.

        along (int, optional): 
            Dimension along which the ``value`` is to be added, if ``value`` is a 1-dimensional array.
    """
    def __init__(self, seed, value, along = 0L):
        allzero = True

        if isinstance(value, collections.Sequence):
            if len(value) != seed.shape[along]:
                raise ValueError("length of sequence-like 'value' should equal seed dimension in 'dim'")
            for x in value:
                if x:
                    allzero = False
        else:
            allzero = (value == 0)

        self.__seed = seed
        self.__value = value
        self.__along = along
        self.__allzero = allzero

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__seed.shape

    @property
    def sparse(self) -> bool:
        if self.__seed.sparse:
            return self.__allzero
        else:
            return False

    def extract_dense_array(self, idx):
        base = extract_dense_array(self.__seed, idx)

        if isinstance(self.__value, collections.Sequence):
            value = self.__value

            curslice = idx[self.__along]
            if curslice:
                value = value[curslice]

            if isinstance(self.__value, numpy.ndarray) and self.__along == len(base.shape):
                base += value
            else:
                # brain too smooth to figure out how to get numpy to do this quickly for me.
                contents = [slice(None)] * len(base.shape)
                for i in range(len(value)):
                    contents[self.__along] = i
                    base[*contents] += value[i]

        else:
            base += self.__value

        return base

    def extract_sparse_array(self, idx):
        # Assumed to be sparse at this point, which only occurs when value is all-zero,
        # i.e., it's a no-op. Indices are also assumed to be unique.
        return self.__seed.extract_sparse_array(idx)
