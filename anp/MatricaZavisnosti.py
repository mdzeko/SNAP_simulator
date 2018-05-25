import numpy as np
from numba import jitclass, float64


spec = [
    ('Z', float64[:, :]),
    ('kombinacija', float64[:]),
]


@jitclass(spec)
class MatricaZavisnosti(object):

    def __init__(self, np_matrix, kombinacija=None):
        self.Z = np.array(np_matrix)
        self.kombinacija = kombinacija
