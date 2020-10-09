import numpy as np


class SymmetricBilinearForm(object):
    def __init__(self, abc):
        """
        abc: (uv) => (A,B,C)
            A vectorized function that produces
            the coefficents of the form
        """
        self.abc = abc

    def matrix(self, uv):
        A, B, C = self.abc(uv)

        # fmt: off
        return np.array(
            [[A, B], 
            [B, C]]
        ).swapaxes(0, 2).swapaxes(1, 2)
        # fmt: on
