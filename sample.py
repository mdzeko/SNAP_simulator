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
import time

totalElapsed = time.process_time()
brojKlastera = 1
brojKriterija = 5

t1 = np.array([])
t2 = np.array([])
t3 = np.array([])
t4 = np.array([])
t5 = np.array([])
t_res = np.array([])

gen = Generator(brojKlastera, brojKriterija)
header = ['ANP1', 'ANP2', 'ANP3', 'ANP4', 'SNAP1', 'min_ANP', 'max_ANP',
          'R1_min', 'R1_max', 'SNAP1_elem_R1',
          'R2_min', 'R2_max', 'SNAP_elem_R2',
          'R3_min', 'R3_max', 'SNAP_elem_R3',
          'rank_ANP1', 'rank_ANP2', 'rank_ANP3', 'rank_ANP4', 'rank_SNAP1']
df = pd.DataFrame(columns=header)
with open("test.csv", "r") as csvUsporedbe, open("zavisnosti.csv", "r") as csvZavisnosti:
    usporedbeReader = csv.reader(csvUsporedbe, delimiter=';')
    counter = 0
    for usporedba in usporedbeReader:
        counter += 1
        print(counter, "/10")
        usp = MatricaUsporedbi(gen.izradiMatUsporedbe(ast.literal_eval(usporedba[0]), 4))
        zavisnostiReader = csv.reader(csvZavisnosti, delimiter=';')
        csvZavisnosti.seek(0)
        for zavisnost in zavisnostiReader:
            start_inner = time.process_time()
            if np.sum(np.array(ast.literal_eval(zavisnost[0]))) == 0:
                continue
            zav = MatricaZavisnosti(gen.izradiMatZavisnosti(ast.literal_eval(zavisnost[0]), 4))

            startAnp1 = time.process_time()
            anp1 = ANP(usp.weights, zav.Z)
            anp1.simulate()
            t1 = np.append(t1, time.process_time() - startAnp1)

            startAnp2 = time.process_time()
            anp2 = ANP(usp.weights, zav.Z)
            anp2.simulate(fiktivnaAlt=True, matricaPrijelaza=False)
            t2 = np.append(t2, time.process_time() - startAnp2)

            startAnp3 = time.process_time()
            anp3 = ANP(usp.weights, zav.Z)
            anp3.simulate(fiktivnaAlt=False, matricaPrijelaza=True)
            t3 = np.append(t3, time.process_time() - startAnp3)

            startAnp4 = time.process_time()
            anp4 = ANP(usp.weights, zav.Z)
            anp4.simulate(matricaPrijelaza=True, fiktivnaAlt=True)
            t4 = np.append(t4, time.process_time() - startAnp4)

            startSnap = time.process_time()
            snap = SNAP(usp.weights, zav.Z)
            snap.simulate()
            t5 = np.append(t5, time.process_time() - startSnap)

            startResults = time.process_time()
            anp_array = np.hstack((anp1.tezine, anp2.tezine, anp3.tezine, anp4.tezine))
            min_anp = np.amin(anp_array, axis=1)
            max_anp = np.amax(anp_array, axis=1)

            r1_min = min_anp * 0.9
            r1_max = max_anp * 1.1
            snap_elem_r1 = np.average(np.logical_and(snap.tezine_S1 <= r1_max, snap.tezine_S1 >= r1_min))

            r2_min = (min_anp - 0.05).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
            r2_max = max_anp + 0.05
            snap_elem_r2 = np.average(np.logical_and(snap.tezine_S1 <= r2_max, snap.tezine_S1 >= r2_min))

            r3_min = (min_anp - 0.1).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
            r3_max = max_anp + 0.1
            snap_elem_r3 = np.sum(np.logical_and(snap.tezine_S1 <= r3_max, snap.tezine_S1 >= r3_min))

            rank_anp1 = rankdata(anp1.tezine, method='ordinal')
            rank_anp2 = rankdata(anp2.tezine, method='ordinal')
            rank_anp3 = rankdata(anp3.tezine, method='ordinal')
            rank_anp4 = rankdata(anp4.tezine, method='ordinal')
            rank_snap1 = rankdata(snap.tezine_S1, method='ordinal')

            df = df.append(pd.DataFrame([{'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
                                          'ANP3': anp3.tezine.flatten(), 'ANP4': anp4.tezine.flatten(),
                                          'SNAP1': snap.tezine_S1.flatten(), 'min_ANP': min_anp, 'max_ANP': max_anp,
                                          'R1_min': r1_min, 'R1_max': r1_max, 'SNAP1_elem_R1': snap_elem_r1,
                                          'R2_min': r2_min, 'R2_max': r2_max, 'SNAP_elem_R2': snap_elem_r2,
                                          'R3_min': r3_min, 'R3_max': r3_max, 'SNAP_elem_R3': snap_elem_r3,
                                          'rank_ANP1': rank_anp1, 'rank_ANP2': rank_anp2, 'rank_ANP3': rank_anp3,
                                          'rank_ANP4': rank_anp4, 'rank_SNAP1': rank_snap1}]))
            t_res = np.append(t_res, time.process_time() - startResults)

print("Ukupno vrijeme izvršavanja", time.process_time() - totalElapsed)
print("ANP 1")
print("max", np.max(t1))
print("min", np.min(t1))
print("prosjek", np.mean(t1))
print("medijan", np.median(t1))
print("=================")
print("ANP 2")
print("max", np.max(t2))
print("min", np.min(t2))
print("prosjek", np.mean(t2))
print("medijan", np.median(t2))
print("=================")
print("ANP 3")
print("max", np.max(t3))
print("min", np.min(t3))
print("prosjek", np.mean(t3))
print("medijan", np.median(t3))
print("=================")
print("ANP 4")
print("max", np.max(t4))
print("min", np.min(t4))
print("prosjek", np.mean(t4))
print("medijan", np.median(t4))
print("=================")
print("SNAP")
print("max", np.max(t5))
print("min", np.min(t5))
print("prosjek", np.mean(t5))
print("medijan", np.median(t5))
print("=================")
print("Operacija izračuna rezultata")
print("max", np.max(t_res))
print("min", np.min(t_res))
print("prosjek", np.mean(t_res))
print("medijan", np.median(t_res))

df = pd.DataFrame()
df['ANP1'] = t1
df['ANP2'] = t2,
df['ANP3'] = t3,
df['ANP4'] = t4,
df['SNAP1'] = t5,
df['rezultati'] = t_res
df.to_csv("mjere.csv")

# usp = MatricaUsporedbi(gen.izradiMatUsporedbe([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], brojKriterija))
# zav = MatricaZavisnosti(
#     gen.izradiMatZavisnosti([4, 1, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 4, 2, 2, 2, 3, 3, 3], brojKriterija))
# snap = SNAP(usp, zav)
# snap.simulate()
# snap.printResults()

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
