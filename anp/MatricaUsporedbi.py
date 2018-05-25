import numpy as np
from numba import jitclass, float64, int32, boolean

spec = [
    ('U', float64[:, :]),
    ('konzistentnost', float64),
    ('relevantna', boolean),
    ('kombinacija', float64[:]),
    ('sumCols', float64[:]),
    ('X', float64[:, :]),
    ('weights', float64[:]),
    ('Y', float64[:, :]),
    ('sumRows', float64[:]),
    ('sumRowsDividedByWeights', float64[:]),
    ('lam', float64),
    ('CI', float64),
]


@jitclass(spec)
class MatricaUsporedbi(object):

    def __init__(self, np_matrix, kombinacija=None):
        self.U = np.array(np_matrix)
        self.konzistentnost = self.izracunajKonzistentnost()
        self.relevantna = self.konzistentnost <= 0.1
        self.kombinacija = kombinacija
        self.sumCols = np.array
        self.X = np.array
        self.weights = np.array
        self.Y = np.array
        self.sumRows = np.array
        self.sumRowsDividedByWeights = np.array
        self.lam = 0
        self.CI = 0

    def ispisiMatricu(self):
        print('A \n', self.U)
        print('SUM:', self.sumCols)
        print('X: \n', self.X)
        print('Tezine: \n', self.weights)
        print('Y: \n', self.Y)
        print('Zbroj/Tezine: \n', self.sumRowsDividedByWeights)
        print('Lambda: ', self.lam)
        print('CI: ', self.CI)
        print('CO: ', self.konzistentnost)

    def izracunajKonzistentnost(self):
        constants = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.4, 9: 1.45, 10: 1.49}
        self.izracunajSumuStupaca()
        self.normalizirajStupceSumom()
        self.weights = self.X.mean(axis=1, keepdims=True)

        self.Y = np.array(self.U, copy=True)
        for i in range(0, self.weights.size):
            self.Y[:, i] *= self.weights.item(i)

        self.sumRows = np.array(np.sum(self.Y, axis=1))

        n = len(self.sumRows)
        temp = self.sumRows.reshape(n, 1)
        self.sumRowsDividedByWeights = temp / self.weights

        self.lam = self.sumRowsDividedByWeights.mean()

        self.CI = (self.lam - n) / (n - 1)
        CO = self.CI / (constants[n])
        return CO

    def normalizirajStupceSumom(self):
        self.X = np.array(self.U, copy=True)
        for i in range(0, self.sumCols.size):
            self.X[:, i] /= self.sumCols.item(i)

    def izracunajSumuStupaca(self):
        self.sumCols = np.array(np.sum(self.U, axis=0))
