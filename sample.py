from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
import numpy as np
from scipy.stats import rankdata
from functools import partial
import csv
import ast
import time
import pandas as pd
import sys
from multiprocessing import Pool as ThreadPool

totalElapsed = time.process_time()
brojKlastera = 1
brojKriterija = 4

header = ['ANP1', 'ANP2', 'ANP3', 'ANP4', 'SNAP1', 'min_ANP', 'max_ANP', 'R1_min', 'R1_max',
          'SNAP1_elem_R1', 'SNAP2_elem_R1', 'SNAP3_elem_R1', 'SNAP4_elem_R1',
          'SNAP5_elem_R1', 'SNAP6_elem_R1', 'SNAP7_elem_R1', 'SNAP8_elem_R1',
          'SNAP9_elem_R1', 'SNAP10_elem_R1', 'SNAP11_elem_R1', 'SNAP12_elem_R1',
          'R2_min', 'R2_max',
          'SNAP1_elem_R2', 'SNAP2_elem_R2', 'SNAP3_elem_R2', 'SNAP4_elem_R2',
          'SNAP5_elem_R2', 'SNAP6_elem_R2', 'SNAP7_elem_R2', 'SNAP8_elem_R2',
          'SNAP9_elem_R2', 'SNAP10_elem_R2', 'SNAP11_elem_R2', 'SNAP12_elem_R2',
          'R3_min', 'R3_max',
          'SNAP1_elem_R3', 'SNAP2_elem_R3', 'SNAP3_elem_R3', 'SNAP4_elem_R3',
          'SNAP5_elem_R3', 'SNAP6_elem_R3', 'SNAP7_elem_R3', 'SNAP8_elem_R3',
          'SNAP9_elem_R3', 'SNAP10_elem_R3', 'SNAP11_elem_R3', 'SNAP12_elem_R3',
          'rank_ANP1', 'rank_ANP2', 'rank_ANP3', 'rank_ANP4',
          'rank_SNAP1', 'rank_SNAP2', 'rank_SNAP3', 'rank_SNAP4',
          'rank_SNAP5', 'rank_SNAP6', 'rank_SNAP7', 'rank_SNAP8',
          'rank_SNAP9', 'rank_SNAP10', 'rank_SNAP11', 'rank_SNAP12']

correlationColumns = ['SNAP1_elem_R1', 'SNAP2_elem_R1', 'SNAP3_elem_R1', 'SNAP4_elem_R1',
                      'SNAP5_elem_R1', 'SNAP6_elem_R1', 'SNAP7_elem_R1', 'SNAP8_elem_R1',
                      'SNAP9_elem_R1', 'SNAP10_elem_R1', 'SNAP11_elem_R1', 'SNAP12_elem_R1',
                      'SNAP1_elem_R2', 'SNAP2_elem_R2', 'SNAP3_elem_R2', 'SNAP4_elem_R2',
                      'SNAP5_elem_R2', 'SNAP6_elem_R2', 'SNAP7_elem_R2', 'SNAP8_elem_R2',
                      'SNAP9_elem_R2', 'SNAP10_elem_R2', 'SNAP11_elem_R2', 'SNAP12_elem_R2',
                      'SNAP1_elem_R3', 'SNAP2_elem_R3', 'SNAP3_elem_R3', 'SNAP4_elem_R3',
                      'SNAP5_elem_R3', 'SNAP6_elem_R3', 'SNAP7_elem_R3', 'SNAP8_elem_R3',
                      'SNAP9_elem_R3', 'SNAP10_elem_R3', 'SNAP11_elem_R3', 'SNAP12_elem_R3']


def printExecutionTimes(t1, t2, t3, t4, t5, t_res):
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
    return


def doSimulation(usporedba, zavisnost):
    counter = 0
    usp = MatricaUsporedbi(Generator.izradiMatUsporedbe(ast.literal_eval(usporedba), brojKriterija))
    zav = MatricaZavisnosti(Generator.izradiMatZavisnosti(ast.literal_eval(zavisnost), brojKriterija))

    anp1 = ANP(usp.weights, zav.Z)
    anp1.simulate()

    anp2 = ANP(usp.weights, zav.Z)
    anp2.simulate(fiktivnaAlt=True, matricaPrijelaza=False)

    anp3 = ANP(usp.weights, zav.Z)
    anp3.simulate(fiktivnaAlt=False, matricaPrijelaza=True)

    anp4 = ANP(usp.weights, zav.Z)
    anp4.simulate(matricaPrijelaza=True, fiktivnaAlt=True)

    snap = SNAP(usp.weights, zav.Z)
    snap.simulate()

    anp_array = np.hstack((anp1.tezine, anp2.tezine, anp3.tezine, anp4.tezine))
    min_anp = np.amin(anp_array, axis=1)
    max_anp = np.amax(anp_array, axis=1)

    r1_min = min_anp * 0.9
    r1_max = max_anp * 1.1
    snap_elem_r1 = np.average(np.logical_and(snap.tezine_S1 <= r1_max, snap.tezine_S1 >= r1_min))
    snap2_elem_r1 = np.average(np.logical_and(snap.tezine_S2 <= r1_max, snap.tezine_S2 >= r1_min))
    snap3_elem_r1 = np.average(np.logical_and(snap.tezine_S3 <= r1_max, snap.tezine_S3 >= r1_min))
    snap4_elem_r1 = np.average(np.logical_and(snap.tezine_S4 <= r1_max, snap.tezine_S4 >= r1_min))
    snap5_elem_r1 = np.average(np.logical_and(snap.tezine_S5 <= r1_max, snap.tezine_S5 >= r1_min))
    snap6_elem_r1 = np.average(np.logical_and(snap.tezine_S6 <= r1_max, snap.tezine_S6 >= r1_min))
    snap7_elem_r1 = np.average(np.logical_and(snap.tezine_S7 <= r1_max, snap.tezine_S7 >= r1_min))
    snap8_elem_r1 = np.average(np.logical_and(snap.tezine_S8 <= r1_max, snap.tezine_S8 >= r1_min))
    snap9_elem_r1 = np.average(np.logical_and(snap.tezine_S9 <= r1_max, snap.tezine_S9 >= r1_min))
    snap10_elem_r1 = np.average(np.logical_and(snap.tezine_S10 <= r1_max, snap.tezine_S10 >= r1_min))
    snap11_elem_r1 = np.average(np.logical_and(snap.tezine_S11 <= r1_max, snap.tezine_S11 >= r1_min))
    snap12_elem_r1 = np.average(np.logical_and(snap.tezine_S12 <= r1_max, snap.tezine_S12 >= r1_min))

    r2_min = (min_anp - 0.05).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
    r2_max = max_anp + 0.05
    snap_elem_r2 = np.average(np.logical_and(snap.tezine_S1 <= r2_max, snap.tezine_S1 >= r2_min))
    snap2_elem_r2 = np.average(np.logical_and(snap.tezine_S2 <= r2_max, snap.tezine_S2 >= r2_min))
    snap3_elem_r2 = np.average(np.logical_and(snap.tezine_S3 <= r2_max, snap.tezine_S3 >= r2_min))
    snap4_elem_r2 = np.average(np.logical_and(snap.tezine_S4 <= r2_max, snap.tezine_S4 >= r2_min))
    snap5_elem_r2 = np.average(np.logical_and(snap.tezine_S5 <= r2_max, snap.tezine_S5 >= r2_min))
    snap6_elem_r2 = np.average(np.logical_and(snap.tezine_S6 <= r2_max, snap.tezine_S6 >= r2_min))
    snap7_elem_r2 = np.average(np.logical_and(snap.tezine_S7 <= r2_max, snap.tezine_S7 >= r2_min))
    snap8_elem_r2 = np.average(np.logical_and(snap.tezine_S8 <= r2_max, snap.tezine_S8 >= r2_min))
    snap9_elem_r2 = np.average(np.logical_and(snap.tezine_S9 <= r2_max, snap.tezine_S9 >= r2_min))
    snap10_elem_r2 = np.average(np.logical_and(snap.tezine_S10 <= r2_max, snap.tezine_S10 >= r2_min))
    snap11_elem_r2 = np.average(np.logical_and(snap.tezine_S11 <= r2_max, snap.tezine_S11 >= r2_min))
    snap12_elem_r2 = np.average(np.logical_and(snap.tezine_S12 <= r2_max, snap.tezine_S12 >= r2_min))

    r3_min = (min_anp - 0.1).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
    r3_max = max_anp + 0.1
    snap_elem_r3 = np.average(np.logical_and(snap.tezine_S1 <= r3_max, snap.tezine_S1 >= r3_min))
    snap2_elem_r3 = np.average(np.logical_and(snap.tezine_S2 <= r3_max, snap.tezine_S2 >= r3_min))
    snap3_elem_r3 = np.average(np.logical_and(snap.tezine_S3 <= r3_max, snap.tezine_S3 >= r3_min))
    snap4_elem_r3 = np.average(np.logical_and(snap.tezine_S4 <= r3_max, snap.tezine_S4 >= r3_min))
    snap5_elem_r3 = np.average(np.logical_and(snap.tezine_S5 <= r3_max, snap.tezine_S5 >= r3_min))
    snap6_elem_r3 = np.average(np.logical_and(snap.tezine_S6 <= r3_max, snap.tezine_S6 >= r3_min))
    snap7_elem_r3 = np.average(np.logical_and(snap.tezine_S7 <= r3_max, snap.tezine_S7 >= r3_min))
    snap8_elem_r3 = np.average(np.logical_and(snap.tezine_S8 <= r3_max, snap.tezine_S8 >= r3_min))
    snap9_elem_r3 = np.average(np.logical_and(snap.tezine_S9 <= r3_max, snap.tezine_S9 >= r3_min))
    snap10_elem_r3 = np.average(np.logical_and(snap.tezine_S10 <= r3_max, snap.tezine_S10 >= r3_min))
    snap11_elem_r3 = np.average(np.logical_and(snap.tezine_S11 <= r3_max, snap.tezine_S11 >= r3_min))
    snap12_elem_r3 = np.average(np.logical_and(snap.tezine_S12 <= r3_max, snap.tezine_S12 >= r3_min))

    rank_anp1 = rankdata(anp1.tezine, method='ordinal')
    rank_anp2 = rankdata(anp2.tezine, method='ordinal')
    rank_anp3 = rankdata(anp3.tezine, method='ordinal')
    rank_anp4 = rankdata(anp4.tezine, method='ordinal')
    rank_snap1 = rankdata(snap.tezine_S1, method='ordinal')
    rank_snap2 = rankdata(snap.tezine_S2, method='ordinal')
    rank_snap3 = rankdata(snap.tezine_S3, method='ordinal')
    rank_snap4 = rankdata(snap.tezine_S4, method='ordinal')
    rank_snap5 = rankdata(snap.tezine_S5, method='ordinal')
    rank_snap6 = rankdata(snap.tezine_S6, method='ordinal')
    rank_snap7 = rankdata(snap.tezine_S7, method='ordinal')
    rank_snap8 = rankdata(snap.tezine_S8, method='ordinal')
    rank_snap9 = rankdata(snap.tezine_S9, method='ordinal')
    rank_snap10 = rankdata(snap.tezine_S10, method='ordinal')
    rank_snap11 = rankdata(snap.tezine_S11, method='ordinal')
    rank_snap12 = rankdata(snap.tezine_S12, method='ordinal')

    counter += 1
    res = ((zavisnost, usporedba),
           {'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
            'ANP3': anp3.tezine.flatten(), 'ANP4': anp4.tezine.flatten(),
            'SNAP1': snap.tezine_S1.flatten(), 'min_ANP': min_anp, 'max_ANP': max_anp,
            'R1_min': r1_min, 'R1_max': r1_max, 'SNAP1_elem_R1': snap_elem_r1,
            'SNAP2_elem_R1': snap2_elem_r1, 'SNAP3_elem_R1': snap3_elem_r1,
            'SNAP4_elem_R1': snap4_elem_r1, 'SNAP5_elem_R1': snap5_elem_r1,
            'SNAP6_elem_R1': snap6_elem_r1, 'SNAP7_elem_R1': snap7_elem_r1,
            'SNAP8_elem_R1': snap8_elem_r1, 'SNAP9_elem_R1': snap9_elem_r1,
            'SNAP10_elem_R1': snap10_elem_r1, 'SNAP11_elem_R1': snap11_elem_r1,
            'SNAP12_elem_R1': snap12_elem_r1,
            'R2_min': r2_min, 'R2_max': r2_max, 'SNAP1_elem_R2': snap_elem_r2,
            'SNAP2_elem_R2': snap2_elem_r2, 'SNAP3_elem_R2': snap3_elem_r2,
            'SNAP4_elem_R2': snap4_elem_r2, 'SNAP5_elem_R2': snap5_elem_r2,
            'SNAP6_elem_R2': snap6_elem_r2, 'SNAP7_elem_R2': snap7_elem_r2,
            'SNAP8_elem_R2': snap8_elem_r2, 'SNAP9_elem_R2': snap9_elem_r2,
            'SNAP10_elem_R2': snap10_elem_r2, 'SNAP11_elem_R2': snap11_elem_r2,
            'SNAP12_elem_R2': snap12_elem_r2,
            'R3_min': r3_min, 'R3_max': r3_max, 'SNAP1_elem_R3': snap_elem_r3,
            'SNAP2_elem_R3': snap2_elem_r3, 'SNAP3_elem_R3': snap3_elem_r3,
            'SNAP4_elem_R3': snap4_elem_r3, 'SNAP5_elem_R3': snap5_elem_r3,
            'SNAP6_elem_R3': snap6_elem_r3, 'SNAP7_elem_R3': snap7_elem_r3,
            'SNAP8_elem_R3': snap8_elem_r3, 'SNAP9_elem_R3': snap9_elem_r3,
            'SNAP10_elem_R3': snap10_elem_r3, 'SNAP11_elem_R3': snap11_elem_r3,
            'SNAP12_elem_R3': snap12_elem_r3,
            'rank_ANP1': rank_anp1, 'rank_ANP2': rank_anp2, 'rank_ANP3': rank_anp3,
            'rank_ANP4': rank_anp4, 'rank_SNAP1': rank_snap1, 'rank_SNAP2': rank_snap2,
            'rank_SNAP3': rank_snap3, 'rank_SNAP4': rank_snap4, 'rank_SNAP5': rank_snap5,
            'rank_SNAP6': rank_snap6, 'rank_SNAP7': rank_snap7, 'rank_SNAP8': rank_snap8,
            'rank_SNAP9': rank_snap9, 'rank_SNAP10': rank_snap10, 'rank_SNAP11': rank_snap11,
            'rank_SNAP12': rank_snap12})
    return res


def main():
    pool = ThreadPool(4)
    # execute only if run as a script
    with open("test.csv", "r") as csvUsporedbe, open("test_z.csv", "r") as csvZavisnosti:
        usporedbeReader = csv.reader(csvUsporedbe, delimiter=';')
        zavisnostiReader = csv.reader(csvZavisnosti, delimiter=';')
        listaZavisnosti = list(zavisnostiReader)
        listaUsporedbi = list(usporedbeReader)

    counter = 0
    zavLista = []
    for redak in listaZavisnosti:
        zavLista.append(redak[0])
    for usporedba in listaUsporedbi:
        counter += 1
        print(counter)
        doPartOfSimulation = partial(doSimulation, usporedba[0])
        results.update(pool.map(doPartOfSimulation, zavLista))
    # printExecutionTimes(t1, t2, t3, t4, t5, t_res)


def processResults():
    df = pd.DataFrame(list(results.values()))
    print("Broj krit. ", brojKriterija, " broj klast. ", brojKlastera, " broj komb:", len(df.index))
    correlation = df[correlationColumns].corr('spearman')
    distributions = df[correlationColumns].apply(pd.Series.value_counts)
    distributions.to_csv(("distrib_" + str(brojKlastera) + "_" + str(brojKriterija) + ".csv"), header=True, sep=";", na_rep='NaN')
    correlation.to_csv(("corr_" + str(brojKlastera) + "_" + str(brojKriterija) + ".csv"), header=True, sep=";", na_rep='NaN')


if __name__ == "__main__":
    # if sys.argv[1] in ("-h", "-help", "--help") or sys.argv.count() < 2:
    #     print("main.py <datotekaUsporedbi> <datotekaZavisnost> <brojDretvi>")
    #     print("PRIMJER: python main.py '~/usporedbe.csv' '~/zavisnosti.csv' 4")
    #     sys.exit(2)
    # inputUsporedbe, inputZavisnosti, brojDretvi = tuple(sys.argv[1:])
    results = {}
    start = time.time()
    main()
    processResults()
    print(time.time() - start)

# usp = MatricaUsporedbi(gen.izradiMatUsporedbe([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], brojKriterija))
# zav = MatricaZavisnosti(
#     gen.izradiMatZavisnosti([4, 1, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 4, 2, 2, 2, 3, 3, 3], brojKriterija))
# snap = SNAP(usp, zav)
# snap.simulate()
# snap.printResults()

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
