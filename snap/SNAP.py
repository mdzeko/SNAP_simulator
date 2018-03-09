import numpy as np

from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from anp.Supermatrix import Supermatrix


class SNAP:
    tezineUsporedbi = np.array
    zavisnost = MatricaZavisnosti
    CO = np.array
    CI = np.array
    razlike = np.array
    norm = np.array
    norm2 = np.array
    tezine = np.array

    def __init__(self, comparisons: MatricaUsporedbi, dependancies: MatricaZavisnosti):
        self.tezineUsporedbi = np.array(comparisons.weights).reshape((1, comparisons.weights.size))
        self.zavisnost = dependancies

    def simulate(self, writeToLog=False):
        self.CO = np.array(np.sum(self.zavisnost.Z, axis=1))
        self.CI = np.array(np.sum(self.zavisnost.Z, axis=0))
        self.razlike = self.CO - self.CI
        self.norm = self.razlike + np.ptp(self.razlike, axis=0)
        sum = np.sum(self.norm)
        if sum != 0:
            self.norm2 = self.norm / sum
        else:
            self.norm2 = self.norm
        self.tezine = (self.tezineUsporedbi + self.norm2) / 2
