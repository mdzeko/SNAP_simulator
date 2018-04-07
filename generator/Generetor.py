import numpy as np
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
import itertools


class Generator:
    matrices = []
    zmatrices = []

    def __init__(self, numOfClusters: int, numOfCriteria: int):
        """
        :param numOfClusters: Broj klastera za koji se generiraju ulazni parametri
        :param numOfCriteria: Broj kriterija po klasteru
        """
        self.clusters = np.array(range(numOfClusters))
        self.criteria = numOfCriteria

    def generateComparisonMatrix(self):
        return np.matrix('1 2 2 2; 0.5 1 1 1; 0.5 1 1 1; 0.5 1 1 1')

    def generateAllComparisonMatrices(self, writeToFile=True):
        NUM_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9]
        if writeToFile:
            fh = open("usporedbe.txt", "w")
            fh_prezivjele = open("prezivjele.txt", "w")

        n = self.criteria  # red kvadratne matrice
        br = int((n - 1) * n / 2)  # broj elemenata u gornjem trokutu
        gen_array = list(itertools.combinations_with_replacement(NUM_POOL, br))
        for combination in gen_array:
            a = np.array(list(combination))
            b = 1 / a
            U = MatricaUsporedbi(np.matrix(self.izradiMatricu(self, a, b, n)), a)
            if U.relevantna:
                self.matrices.append(U)
                if writeToFile:
                    fh_prezivjele.write(str(a.tolist() + [' konz:', U.konzistentnost]) + '\n')
            if writeToFile:
                fh.write(str(a.tolist() + [' konz:', U.konzistentnost]) + '\n')
        if writeToFile:
            fh.close()
            fh_prezivjele.close()

    def generateDependancyMatrices(self, writeToFile=True):
        if writeToFile:
            fh = open("zavisnosti.txt", "w")

        n = self.criteria  # red kvadratne matrice
        br = int((n - 1) * n)  # zbroj broja elemenata u gornjem i donjem trokutu
        gen_array = list(itertools.combinations_with_replacement(range(0, 5), br))
        for combination in gen_array:
            if writeToFile:
                fh.write(str(list(combination)) + '\n')
            pola = int(br / 2)
            a = np.array(list(combination[:pola]))
            b = np.array(list(combination[pola:]))
            Z = MatricaZavisnosti(np.matrix(self.izradiMatricu(a, b, n, 0)), combination)
            self.zmatrices.append(Z)
        if writeToFile:
            fh.close()

    @staticmethod
    def izradiMatricu(self, gornji_trokut, donji_trokut, n, dijagonala=1):

        if dijagonala == 0:
            uper = np.zeros((n, n))
        else:
            uper = np.identity(n)

        uper[np.triu_indices(n, 1)] = gornji_trokut
        indeksi = np.tril_indices(n, -1)
        uskladi_indekse = list(zip(indeksi[0], indeksi[1]))
        uskladi_indekse.sort(key=lambda x: x[1])
        indeksi_DT = (
            np.array(list(map(lambda x: x[0], uskladi_indekse))), np.array(list(map(lambda x: x[1], uskladi_indekse))))
        uper[indeksi_DT] = donji_trokut
        return uper
