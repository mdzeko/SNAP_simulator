import numpy as np
from anp.MatricaUsporedbi import MatricaUsporedbi
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
        NUM_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        n = self.criteria  # red kvadratne matrice
        br = int((n - 1) * n / 2)  # broj elemenata u gornjem trokutu
        if writeToFile:
            with open("prezivjele_" + str(self.criteria) + ".csv", "w") as csvPrezivjele:
                for combination in itertools.combinations_with_replacement(NUM_POOL, br):
                    a = np.array(list(combination))
                    b = np.divide(1., a.astype(float), out=np.zeros_like(a.astype(float)), where=a != 0)
                    # b = 1 / a
                    U = MatricaUsporedbi(np.matrix(self.izradiMatricu(a, b, n)), a)
                    if U.relevantna:
                        if writeToFile:
                            csvPrezivjele.write(str(a.tolist()) + ";" + str(U.konzistentnost) + "\n")

    def generateDependancyMatrices(self, writeToFile=True):
        if writeToFile:
            fh = open("zavisnosti_" + str(len(self.clusters)) + "_" + str(self.criteria) + ".csv", "w")
        n = self.criteria  # red kvadratne matrice
        br = int((n - 1) * n)  # zbroj broja elemenata u gornjem i donjem trokutu
        gen_array = list(itertools.combinations_with_replacement(range(0, 5), br))
        for combination in gen_array:
            if writeToFile:
                fh.write(str(list(combination)) + "\n")
            # pola = int(br / 2)
            # a = np.array(list(combination[:pola]))
            # b = np.array(list(combination[pola:]))
            # Z = MatricaZavisnosti(np.matrix(self.izradiMatricu(a, b, n, 0)), combination)
        if writeToFile:
            fh.close()

    @staticmethod
    def izradiMatricu(gornji_trokut, donji_trokut, n, dijagonala=1):
        if dijagonala == 0:
            uper = np.zeros((n, n))
        else:
            uper = np.identity(n)
        try:
            uper[np.triu_indices(n, 1)] = gornji_trokut
            indeksi = np.tril_indices(n, -1)
            uskladi_indekse = list(zip(indeksi[0], indeksi[1]))
            uskladi_indekse.sort(key=lambda x: x[1])
            indeksi_DT = (
                np.array(list(map(lambda x: x[0], uskladi_indekse))),
                np.array(list(map(lambda x: x[1], uskladi_indekse))))
            uper[indeksi_DT] = donji_trokut
        except ValueError:
            print(gornji_trokut)
            print(donji_trokut)
            print(n)
            print(dijagonala)

        return uper

    @staticmethod
    def izradiMatUsporedbe(kombinacija, n):
        a = np.array(kombinacija)
        a = a / 1
        b = np.divide(1., a, out=np.zeros_like(a), where=a != 0.0)
        return Generator.izradiMatricu(a, b, n)

    @staticmethod
    def izradiMatZavisnosti(kombinacija, n):
        pola = int(len(kombinacija) / 2)
        a = np.array(list(kombinacija[:pola]))
        b = np.array(list(kombinacija[pola:]))
        return Generator.izradiMatricu(a, b, n, 0)
