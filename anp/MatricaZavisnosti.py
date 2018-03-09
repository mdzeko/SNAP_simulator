import numpy as np


class MatricaZavisnosti(object):
    Z = np.matrix
    kombinacija = np.array
    def __init__(self, np_matrix: np.matrix, kombinacija):
        self.Z = np_matrix
        self.kombinacija = kombinacija