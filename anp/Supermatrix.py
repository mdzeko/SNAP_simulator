import numpy as np


class Supermatrix:
    S = np.array
    Z = np.array
    comparisonWeights = np.array
    L = np.array
    konvergira = 0
    teziNuli = 0
    cezarova = 0

    def __init__(self, tezine, zavisnosti):
        self.comparisonWeights = np.array(tezine)
        self.Z = np.array(zavisnosti, dtype=np.float64)
        self.calculateSupermatrix()

    def calculateSupermatrix(self):
        self.S = np.array
        sums = self.Z.sum(axis=0)
        self.S = np.divide(self.Z, sums, out=np.zeros_like(self.Z), where=sums!=0)
        self.S = np.hstack((self.comparisonWeights, self.S))
        retci, stupci = self.S.shape
        self.S = np.vstack((np.zeros(stupci), self.S))

    def calculateLimitMatrix(self, iterNumForCesarSum):
        self.L = np.array(self.S, dtype=np.float64)
        np.matmul(self.S, self.S, self.L)
        counter = 0
        while True:
            if self.allColumnsClose(self.L):
                # alternateMat = np.zeros(self.L.shape)
                # print("stupci su jednaki - broj koraka ", counter, "\n", self.L)
                self.konvergira = 1
                break
                # np.matmul(self.L, self.L, alternateMat)
                # if np.allclose(alternateMat, self.L):
                #     break
                # else:
                #     self.doAverageMatrix(self.comparisonWeights.shape[0], alternateMat)
                #     break
            np.matmul(self.L, self.L, self.L)
            if not self.checkSumIsOk(np.array(np.sum(self.L, axis=0))):
                self.normalize(self.L)
            counter += 1
            if counter == 30:
                self.cezarova = 1
                self.doAverageMatrix(iterNumForCesarSum, self.L)
                break

    def doAverageMatrix(self, iterNum, matrix: np.array):
        sumMatrix = np.array(matrix, dtype=np.float64)
        for i in range(iterNum):
            np.matmul(matrix, matrix, matrix)
            sumMatrix += matrix
        return sumMatrix / iterNum

    def getCriteraWieghts(self):
        return np.array(self.L[1:, 0])

    def allColumnsClose(self, matrix: np.array):
        for row in matrix.T:
            if np.count_nonzero(row) > 0 and not np.allclose(matrix.T[0], row, rtol=1e-05, atol=1e-08):
                return False
        return True

    def checkSumIsOk(self, sums: np.array):
        for i in range(0, len(sums)-1):
            if sums[i] != 0 and sums[i] < .99999999999:
                self.teziNuli = 1
                return False
        return True

    def normalize(self, matrix: np.array):
        sumCols = np.array(np.sum(matrix, axis=0))
        for i in range(0, sumCols.size):
            if sumCols[i] == 0:
                continue
            matrix[:, i] /= sumCols[i]
        return matrix