import numpy as np


class MatricaZavisnosti(object):
    Z = np.array
    kombinacija = np.array

    def __init__(self, np_matrix, kombinacija=None):
        self.Z = np.array(np_matrix)
        self.kombinacija = kombinacija
