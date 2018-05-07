from functools import reduce

from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
import numpy as np
from scipy.stats import rankdata
import pandas as pd
import csv
import ast

brojKlastera = 1
brojKriterija = 4
gen = Generator(brojKlastera, brojKriterija)

with open("prezivjele.csv", "r") as csvUsporedbe, open("zavisnosti.csv", "r") as csvZavisnosti:
    usporedbeReader = csv.reader(csvUsporedbe, delimiter=';')
    zavisnostiReader = csv.reader(csvZavisnosti, delimiter=';')
    header = ['ANP1', 'ANP2', 'ANP3', 'ANP4', 'SNAP1', 'min_ANP', 'max_ANP',
              'R1_min', 'R1_max', 'SNAP1_elem_R1',
              'R2_min', 'R2_max', 'SNAP_elem_R2',
              'R3_min', 'R3_max', 'SNAP_elem_R3',
              'rank_ANP1', 'rank_ANP2', 'rank_ANP3', 'rank_ANP4', 'rank_SNAP1']
    df = pd.DataFrame(columns=header)
    for usporedba in usporedbeReader:
        usp = MatricaUsporedbi(Generator.izradiMatUsporedbe(ast.literal_eval(usporedba[0]), 4))
        for zavisnost in zavisnostiReader:
            zav = MatricaZavisnosti(Generator.izradiMatZavisnosti(ast.literal_eval(zavisnost[0]), 4))
            anp1 = ANP(usp, zav)
            anp1.simulate()
            anp2 = ANP(usp, zav)
            anp2.simulate(fiktivnaAlt=True, matricaPrijelaza=False)
            anp3 = ANP(usp, zav)
            anp3.simulate(fiktivnaAlt=False, matricaPrijelaza=True)
            anp4 = ANP(usp, zav)
            anp4.simulate(matricaPrijelaza=True, fiktivnaAlt=True)
            snap = SNAP(usp, zav)
            snap.simulate()

            anp_array = np.hstack((anp1.tezine, anp2.tezine, anp3.tezine, anp4.tezine))
            min_anp = np.amin(anp_array, axis=1)
            max_anp = np.amax(anp_array, axis=1)

            r1_min = min_anp * 0.9
            r1_max = max_anp * 1.1
            snap_elem_r1 = np.average(np.logical_and(snap.tezine <= r1_max, snap.tezine >= r1_min))

            r2_min = (min_anp - 0.05).clip(min=0) # ako razlika daje rezultat ispod nule, postavi na nulu
            r2_max = max_anp + 0.05
            snap_elem_r2 = np.average(np.logical_and(snap.tezine <= r2_max, snap.tezine >= r2_min))

            r3_min = (min_anp - 0.1).clip(min=0) # ako razlika daje rezultat ispod nule, postavi na nulu
            r3_max = max_anp + 0.1
            snap_elem_r3 = np.average(np.logical_and(snap.tezine <= r3_max, snap.tezine >= r3_min))

            rank_anp1 = rankdata(anp1.tezine, method='ordinal')
            rank_anp2 = rankdata(anp2.tezine, method='ordinal')
            rank_anp3 = rankdata(anp3.tezine, method='ordinal')
            rank_anp4 = rankdata(anp4.tezine, method='ordinal')
            rank_snap1 = rankdata(snap.tezine, method='ordinal')

            df = df.append(pd.DataFrame([{'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
                                     'ANP3': anp3.tezine.flatten(), 'ANP4': anp4.tezine.flatten(),
                                     'SNAP1': snap.tezine.flatten(), 'min_ANP': min_anp, 'max_ANP': max_anp,
                                     'R1_min': r1_min, 'R1_max': r1_max, 'SNAP1_elem_R1': snap_elem_r1,
                                     'R2_min': r2_min, 'R2_max': r2_max, 'SNAP_elem_R2': snap_elem_r2,
                                     'R3_min': r3_min, 'R3_max': r3_max, 'SNAP_elem_R3': snap_elem_r3,
                                     'rank_ANP1': rank_anp1, 'rank_ANP2': rank_anp2, 'rank_ANP3': rank_anp3,
                                     'rank_ANP4': rank_anp4, 'rank_SNAP1': rank_snap1}]))





# gen.generateAllComparisonMatrices(writeToFile=True)
# gen.generateDependancyMatrices(writeToFile=True)
# print(gen.izradiMatUsporedbe([1, 1, 1, 1, 1, 1], brojKriterija))
# print(gen.izradiMatZavisnosti([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4], brojKriterija))

# fh = open("rez_%d_klust_%d_krit.txt" % (brojKlastera, brojKriterija), "w")
# fh.write("komb_usporedbe;komb_zavisnosti;konvergira;tezi nuli;cezarova suma;"
#          "ANP_tezine;SNAP_tezine;Razlika;Postotak_razlika_manjih_od_0.1;"+ '\n')
# for usporedba in gen.matrices:
#     for zavisnost in gen.zmatrices:
#         anp = ANP(usporedba, zavisnost)
#         anp.simulate()
#         # anpList.append(anp)
#         snap = SNAP(usporedba, zavisnost)
#         snap.simulate()
#         # snapList.append(snap)
#         razlike = abs(anp.tezine - snap.tezine)
#         brojManjihOd01 = reduce(lambda x, y: x + (1 if y <= 0.1 else 0), razlike, 0)
#         fh.write("%s;%s;%d;%d;%d;%s;%s;%s;%d\n" %
#                  (usporedba.kombinacija, zavisnost.kombinacija,
#                  anp.supermatrica.konvergira, anp.supermatrica.teziNuli, anp.supermatrica.cezarova,
#                   anp.tezine, snap.tezine, razlike, (brojManjihOd01/len(snap.tezine))*100))
#         # S.calculateLimitMatrix()
#         # print(S.L)
# fh.close()
# print("Broj kombinacija za %d klastera i %d kriterija: %d" % (brojKlastera, brojKriterija, len(anpList)))

# U = MatricaUsporedbi(Generator.izradiMatricu(None, [0.2, 0.2, 0.125, 0.125, 0.125, 0.125], [5, 5, 4, 4, 4, 4], 4))
# Z = MatricaZavisnosti(Generator.izradiMatricu(None, [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], 4, 0),
#                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# print("Konzistentnost: ", U.konzistentnost)

# anp = ANP(U, Z)
# anp.simulate()
# snap = SNAP(U, Z)
# snap.simulate()
# print("Supermatrica\n", anp.supermatrica.S)
# print("Supermatrica - gransicna\n", anp.supermatrica.L)
# print("CO", snap.CO)
# print("CI", snap.CI)
# print("CO - CI", snap.razlike)
# print("Norma1", snap.norm)
# print("Norma2", snap.norm2)
# print("Tezine", snap.tezine)

