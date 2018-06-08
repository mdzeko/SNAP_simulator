import numpy as np


class SNAP:
    tezineUsporedbi = np.array
    zavisnost = np.array
    CO = np.array
    CI = np.array
    razlike = np.array

    norm_S1 = np.array
    norm_S3 = np.array
    norm_S5 = np.array
    norm_S7 = np.array
    norm_S9 = np.array
    norm_S11 = np.array

    norm2_S1 = np.array
    norm2_S3 = np.array
    norm2_S5 = np.array
    norm2_S7 = np.array
    norm2_S11 = np.array

    tezine_S2 = np.array
    tezine_S4 = np.array
    tezine_S6 = np.array
    tezine_S8 = np.array
    tezine_S10 = np.array
    tezine_S12 = np.array

    def __init__(self, comparisons, dependancies):
        self.n = comparisons.size
        self.tezineUsporedbi = np.array(comparisons).reshape((1, self.n))
        self.zavisnost = dependancies

    def simulate(self, writeToLog=False):
        # Zbroj redaka
        self.CO = np.array(np.sum(self.zavisnost, axis=1))
        # Zbroj stupaca
        self.CI = np.array(np.sum(self.zavisnost, axis=0))

        # SNAP 7 i 8
        C = self.zavisnost / (np.max(self.CI) + 1)
        D = np.identity(self.n) - C
        E = np.linalg.inv(D)
        F = np.matmul(C, E)
        sumRedaka = np.array(np.sum(F, axis=1))
        sumStupaca = np.array(np.sum(F, axis=0))
        rs = sumRedaka - sumStupaca
        np.around(rs, 8, out=rs)

        # SNAP 9 i 10
        S = self.normalizirajStupceSumom(self.zavisnost, self.CI)
        E = np.ones((self.n, self.n)) / self.n
        G = (0.85 * S) + (0.15 * E)
        G = self.izracunajGranicnuMatricu(G)
        self.norm_S9 = G[0:, 0]

        #SNAP 11 i 12
        H = (0.85 * C) + (0.15 * E)
        I = np.identity(self.n) - H
        J = np.linalg.inv(I)
        K = np.matmul(H, J)
        sumRedaka = np.array(np.sum(K, axis=1))
        sumStupaca = np.array(np.sum(K, axis=0))
        rs2 = sumRedaka - sumStupaca
        np.around(rs2, 8, out=rs2)

        self.razlike = self.CO - self.CI
        self.norm_S1 = self.razlike + np.ptp(self.razlike, axis=0)
        self.norm_S3 = self.razlike + 4 * (self.n - 1)
        self.norm_S5 = self.razlike + abs(min(self.razlike))
        self.norm_S7 = rs + np.ptp(rs, axis=0)
        self.norm_S11 = rs2 + np.ptp(rs2, axis=0)
        sum1 = np.sum(self.norm_S1)
        sum3 = np.sum(self.norm_S3)
        sum5 = np.sum(self.norm_S5)
        sum7 = np.sum(self.norm_S7)
        sum11 = np.sum(self.norm_S11)

        # prvi, treći i peti snap su normalizacija zbrojem
        if sum1 != 0:
            self.norm2_S1 = self.norm_S1 / sum1
        else:
            self.norm2_S1 = np.ones(self.n) / self.n

        if sum3 != 0:
            self.norm2_S3 = self.norm_S3 / sum3
        else:
            self.norm2_S3 = self.norm_S3

        if sum5 != 0:
            self.norm2_S5 = self.norm_S5 / sum5
        else:
            self.norm2_S5 = np.ones(self.n) / self.n

        if sum7 != 0:
            self.norm2_S7 = self.norm_S7 / sum7
        else:
            self.norm2_S7 = np.ones(self.n) / self.n

        if sum11 != 0:
            self.norm2_S11 = self.norm_S11 / sum11
        else:
            self.norm2_S11 = np.ones(self.n) / self.n

        self.tezine_S2 = self.norm2_S1
        self.tezine_S4 = self.norm2_S3
        self.tezine_S6 = self.norm2_S5
        self.tezine_S8 = self.norm2_S7
        self.tezine_S10 = self.norm_S9
        self.tezine_S12 = self.norm2_S11

    def izracunajGranicnuMatricu(self, matrix):
        while True:
            if self.allColumnsClose(matrix):
                break
            np.matmul(matrix, matrix, out=matrix)
        return matrix

    def normalizirajStupceSumom(self, matrix, sumCols):
        X = np.array(matrix, copy=True)
        for i in range(0, sumCols.size):
            X[:, i] = np.divide(X[:, i], sumCols.item(i), out=np.zeros_like(X[:, i]), where=sumCols.item(i) != 0)
        return X

    def allColumnsClose(self, matrix: np.array):
        for row in matrix.T:
            if np.count_nonzero(row) > 0 and not np.allclose(matrix.T[0], row, rtol=1e-05, atol=1e-08):
                return False
        return True

    def printResults(self):
        print(self.tezineUsporedbi)
        print(self.tezine_S2)
        print(self.tezine_S4)
        print(self.tezine_S6)
        print(self.tezine_S8)
        print(self.tezine_S10)
        print(self.tezine_S12)
