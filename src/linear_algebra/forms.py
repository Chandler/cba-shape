class SymmetricBilinearForm(object):
    def __init__(self, abc):
        # (uv) => (A,B,C)
        self.abc = abc

    def matrix(self, uv):
        A, B, C = self.abc(uv)
        return np.array([[A, B], [B, C]]).swapaxes(0, 2).swapaxes(1, 2)
