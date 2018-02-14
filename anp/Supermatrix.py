import numpy as np


class Supermatrix:
    S = np.array
    Z = np.array
    comparisonWeights = np.array
    L = np.array

    def __init__(self, tezine, zavisnosti):
        self.comparisonWeights = np.array(tezine)
        self.Z = np.array(zavisnosti)
        self.calculateSupermatrix()

    def calculateSupermatrix(self):
        if self.Z.sum(axis=0) != 0:
            self.S = np.array(self.Z.tolist()) / self.Z.sum(axis=0)
        else:
            self.S = np.array(self.Z.tolist())
        self.S = np.hstack((self.comparisonWeights, self.S))
        retci, stupci = self.S.shape
        self.S = np.vstack((np.zeros(stupci), self.S))

    def calculateLimitMatrix(self):
        self.L = np.array(self.S)
        np.matmul(self.S, self.S, self.L)
        counter = 0
        while True:
            if (self.L.T == self.L.T[0]).all:
                alternateMat = np.zeros(self.L.shape)
                np.matmul(self.L, self.L, alternateMat)
                if np.allclose(alternateMat, self.L):
                    break
                else:
                    self.doAverageMatrix(self.comparisonWeights.shape[0], alternateMat)
            np.matmul(self.L, self.L, self.L)
            counter += 1
            if counter == 1000:
                break

    def doAverageMatrix(self, numberofcriteria, alternateMat: np.array):
        for i in range(numberofcriteria):
            alternateMat = np.array(alternateMat * alternateMat)
        self.L = alternateMat/numberofcriteria

    def getCriteraWieghts(self):
        return np.array(self.L[1:, 0])
