import numpy as np
from anp.Supermatrix import Supermatrix


class ANP:
    tezineUsporedbi = np.array
    zavisnost = np.array
    tezine = np.array
    supermatrica = Supermatrix
    cesarSumIterNum = 100

    def __init__(self, comparisons, dependancies, iterNumForCesarSum=100):
        self.tezineUsporedbi = np.array(comparisons)
        self.zavisnost = dependancies
        self.cesarSumIterNum = iterNumForCesarSum

    def simulate(self, writeToLog=False, matricaPrijelaza=False, fiktivnaAlt=False):
        self.supermatrica = Supermatrix(self.tezineUsporedbi, self.zavisnost, matPrijelaza=matricaPrijelaza, fiktAlt=fiktivnaAlt)
        self.supermatrica.calculateLimitMatrix(self.cesarSumIterNum)
        self.tezine = self.supermatrica.getCriteraWieghts()

    def printResults(self):
        print("Supermatrica\n", self.supermatrica.S)
        print("Supermatrica - granicna\n", self.supermatrica.L)


