import numpy as np

from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from anp.Supermatrix import Supermatrix


class ANP:
    tezineUsporedbi = np.array
    zavisnost = MatricaZavisnosti
    tezine = np.array
    supermatrica = Supermatrix
    cesarSumIterNum = 100

    def __init__(self, comparisons: MatricaUsporedbi, dependancies: MatricaZavisnosti, iterNumForCesarSum=100):
        self.tezineUsporedbi = np.array(comparisons.weights)
        self.zavisnost = dependancies
        self.cesarSumIterNum = iterNumForCesarSum

    def simulate(self, writeToLog=False, matricaPrijelaza=False, fiktivnaAlt=False):
        self.supermatrica = Supermatrix(self.tezineUsporedbi, self.zavisnost.Z, matPrijelaza=matricaPrijelaza, fiktAlt=fiktivnaAlt)
        self.supermatrica.calculateLimitMatrix(self.cesarSumIterNum)
        self.tezine = self.supermatrica.getCriteraWieghts()

    def printResults(self):
        print("Supermatrica\n", self.supermatrica.S)
        print("Supermatrica - granicna\n", self.supermatrica.L)


