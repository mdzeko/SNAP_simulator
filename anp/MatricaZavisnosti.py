import numpy as np


class MatricaZavisnosti(object):
    Z = np.matrix
    def __init__(self, np_matrix: np.matrix):
        self.Z = np_matrix