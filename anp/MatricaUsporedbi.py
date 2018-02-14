import numpy as np


class MatricaUsporedbi(object):
    relevantna = False
    n = 1
    konzistentnost = 1
    sumCols = np.matrix
    X = np.matrix
    weights = np.matrix
    Y = np.matrix
    sumRows = np.matrix
    sumRowsDividedByWeights = np.matrix
    lam = 0
    constants = {1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.4, 9: 1.45, 10: 1.49}
    CI = 0

    def __init__(self, np_matrix: np.matrix):
        self.U = np_matrix
        self.relevantna = False
        self.konzistentnost = self.izracunajKonzistentnost()

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
        self.sumCols = np.matrix(np.sum(self.U, axis=0))
        self.n = self.sumCols.size
        self.X = np.matrix(data=self.U, copy=True)

        for i in range(0, self.sumCols.size):
            self.X[:, i] /= self.sumCols.item(i)
        self.weights = self.X.mean(1)

        self.Y = np.matrix(data=self.U, copy=True)
        for i in range(0, self.weights.size):
            self.Y[:, i] *= self.weights.item(i)

        self.sumRows = np.matrix(np.sum(self.Y, axis=1))

        self.sumRowsDividedByWeights = self.sumRows / self.weights

        self.lam = self.sumRowsDividedByWeights.mean()

        self.CI = (self.lam - self.n) / (self.n - 1)
        CO = self.CI / (self.constants[self.n])
        self.relevantna = CO <= 0.1
        return CO
