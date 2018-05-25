import numpy as np
from anp.Supermatrix import Supermatrix
from numba import jitclass, float64, int32, deferred_type

supermatrica_type = deferred_type()
spec = [
    ('tezineUsporedbi', float64[:]),
    ('zavisnost', float64[:, :]),
    ('tezine', float64[:]),
    ('supermatrica', supermatrica_type),
    ('cesarSumIterNum', int32)
]
supermatrica_type.define(Supermatrix.class_type.instance_type)


@jitclass(spec)
class ANP(object):
    tezineUsporedbi = np.array
    zavisnost = np.array

    def __init__(self, comparisons, dependancies, iterNumForCesarSum=100):
        self.tezineUsporedbi = np.array(comparisons.weights)
        self.zavisnost = dependancies
        self.tezine = np.array
        self.supermatrica = Supermatrix
        self.cesarSumIterNum = iterNumForCesarSum

    def simulate(self, writeToLog=False, matricaPrijelaza=False, fiktivnaAlt=False):
        self.supermatrica = Supermatrix(self.tezineUsporedbi, self.zavisnost.Z, matPrijelaza=matricaPrijelaza, fiktAlt=fiktivnaAlt)
        self.supermatrica.calculateLimitMatrix(self.cesarSumIterNum)
        self.tezine = self.supermatrica.getCriteraWieghts()

    def printResults(self):
        print("Supermatrica\n", self.supermatrica.S)
        print("Supermatrica - granicna\n", self.supermatrica.L)


