import numpy as np

from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti


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
        # SNAP 3 umjesto linije ispod ima: self.norm = self.razlike + 4 * (n - 1) n je broj kriterija
        # SNAP 5 umjesto linije ispod ima: self.norm = self.razlike + abs(min(razlike)) je broj kriterija
        self.norm = self.razlike + np.ptp(self.razlike, axis=0)
        sum = np.sum(self.norm)
        if sum != 0:
            self.norm2 = self.norm / sum
        else:
            self.norm2 = self.norm

        # Dio iz AHP-a SNAP 1, dvojka, četvorka i šestica i osmica su bez ovoga
        self.tezine = (self.tezineUsporedbi + self.norm2) / 2
        # ============================
        self.tezine = self.tezine.reshape(len(self.tezine.flatten()), 1)

    def printResults(self):
        print("CO", self.CO)
        print("CI", self.CI)
        print("CO - CI", self.razlike)
        print("Norma1", self.norm)
        print("Norma2", self.norm2)
        print("Tezine", self.tezine)
