import ast
import itertools
import math
import sys
import time
from functools import partial
from multiprocessing import Pool as ThreadPool

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP

totalElapsed = time.process_time()
brojKriterija = 4

distributionColumns = ['SNAP2_elem_R1', 'SNAP4_elem_R1', 'SNAP6_elem_R1', 'SNAP8_elem_R1',
                       'SNAP10_elem_R1', 'SNAP12_elem_R1', 'SNAP2_elem_R2', 'SNAP4_elem_R2',
                       'SNAP6_elem_R2', 'SNAP8_elem_R2', 'SNAP10_elem_R2', 'SNAP12_elem_R2',
                       'SNAP2_elem_R3', 'SNAP4_elem_R3', 'SNAP6_elem_R3', 'SNAP8_elem_R3',
                       'SNAP10_elem_R3', 'SNAP12_elem_R3']

correlationColumns = ['sk1_snap2', 'sk2_snap2', 'sk1_snap4', 'sk2_snap4',
                      'sk1_snap6', 'sk2_snap6', 'sk1_snap8', 'sk2_snap8',
                      'sk1_snap10', 'sk2_snap10', 'sk1_snap12', 'sk2_snap12']


def printExecutionTimes(t1, t2, t3, t4, t5, t_res):
    print("Ukupno vrijeme izvrsavanja", time.process_time() - totalElapsed)
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
    print("Operacija izracuna rezultata")
    print("max", np.max(t_res))
    print("min", np.min(t_res))
    print("prosjek", np.mean(t_res))
    print("medijan", np.median(t_res))
    return


def doSimulation(usporedba, zavisnost):
    counter = 0
    usp = MatricaUsporedbi(Generator.izradiMatUsporedbe(usporedba, brojKriterija))
    zav = MatricaZavisnosti(Generator.izradiMatZavisnosti(list(zavisnost), brojKriterija))

    anp1 = ANP(usp.weights, zav.Z)
    anp1.simulate(variant='single')

    anp2 = ANP(usp.weights, zav.Z)
    anp2.simulate(fiktivnaAlt=True, matricaPrijelaza=False, variant='single')

    anp3 = ANP(usp.weights, zav.Z)
    anp3.simulate(fiktivnaAlt=False, matricaPrijelaza=True, variant='single')

    anp4 = ANP(usp.weights, zav.Z)
    anp4.simulate(matricaPrijelaza=True, fiktivnaAlt=True, variant='single')

    snap = SNAP(usp.weights, zav.Z)
    snap.simulate()

    anp_array = np.hstack((anp1.tezine, anp2.tezine, anp3.tezine, anp4.tezine))
    min_anp = np.amin(anp_array, axis=1)
    max_anp = np.amax(anp_array, axis=1)

    r1_min = min_anp * 0.9
    r1_max = max_anp * 1.1
    snap2_elem_r1 = np.average(np.logical_and(snap.tezine_S2 <= r1_max, snap.tezine_S2 >= r1_min))
    snap4_elem_r1 = np.average(np.logical_and(snap.tezine_S4 <= r1_max, snap.tezine_S4 >= r1_min))
    snap6_elem_r1 = np.average(np.logical_and(snap.tezine_S6 <= r1_max, snap.tezine_S6 >= r1_min))
    snap8_elem_r1 = np.average(np.logical_and(snap.tezine_S8 <= r1_max, snap.tezine_S8 >= r1_min))
    snap10_elem_r1 = np.average(np.logical_and(snap.tezine_S10 <= r1_max, snap.tezine_S10 >= r1_min))
    snap12_elem_r1 = np.average(np.logical_and(snap.tezine_S12 <= r1_max, snap.tezine_S12 >= r1_min))

    r2_min = (min_anp - 0.05).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
    r2_max = max_anp + 0.05
    snap2_elem_r2 = np.average(np.logical_and(snap.tezine_S2 <= r2_max, snap.tezine_S2 >= r2_min))
    snap4_elem_r2 = np.average(np.logical_and(snap.tezine_S4 <= r2_max, snap.tezine_S4 >= r2_min))
    snap6_elem_r2 = np.average(np.logical_and(snap.tezine_S6 <= r2_max, snap.tezine_S6 >= r2_min))
    snap8_elem_r2 = np.average(np.logical_and(snap.tezine_S8 <= r2_max, snap.tezine_S8 >= r2_min))
    snap10_elem_r2 = np.average(np.logical_and(snap.tezine_S10 <= r2_max, snap.tezine_S10 >= r2_min))
    snap12_elem_r2 = np.average(np.logical_and(snap.tezine_S12 <= r2_max, snap.tezine_S12 >= r2_min))

    r3_min = (min_anp - 0.1).clip(min=0)  # ako razlika daje rezultat ispod nule, postavi na nulu
    r3_max = max_anp + 0.1
    snap2_elem_r3 = np.average(np.logical_and(snap.tezine_S2 <= r3_max, snap.tezine_S2 >= r3_min))
    snap4_elem_r3 = np.average(np.logical_and(snap.tezine_S4 <= r3_max, snap.tezine_S4 >= r3_min))
    snap6_elem_r3 = np.average(np.logical_and(snap.tezine_S6 <= r3_max, snap.tezine_S6 >= r3_min))
    snap8_elem_r3 = np.average(np.logical_and(snap.tezine_S8 <= r3_max, snap.tezine_S8 >= r3_min))
    snap10_elem_r3 = np.average(np.logical_and(snap.tezine_S10 <= r3_max, snap.tezine_S10 >= r3_min))
    snap12_elem_r3 = np.average(np.logical_and(snap.tezine_S12 <= r3_max, snap.tezine_S12 >= r3_min))

    # Zaokruziti tezine na 5 decimala jer rangiranje zna bit osjetljivo
    rank_anp1 = rankdata(np.round(anp1.tezine, 5), method='ordinal')
    rank_anp2 = rankdata(np.round(anp2.tezine, 5), method='ordinal')
    rank_anp3 = rankdata(np.round(anp3.tezine, 5), method='ordinal')
    rank_anp4 = rankdata(np.round(anp4.tezine, 5), method='ordinal')
    mean_anp_rank = (rank_anp1 + rank_anp2 + rank_anp3 + rank_anp4) / 4
    rank_snap2 = rankdata(np.round(snap.tezine_S2, 5), method='ordinal')
    rank_snap4 = rankdata(np.round(snap.tezine_S4, 5), method='ordinal')
    rank_snap6 = rankdata(np.round(snap.tezine_S6, 5), method='ordinal')
    rank_snap8 = rankdata(np.round(snap.tezine_S8, 5), method='ordinal')
    rank_snap10 = rankdata(np.round(snap.tezine_S10, 5), method='ordinal')
    rank_snap12 = rankdata(np.round(snap.tezine_S12, 5), method='ordinal')

    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap2)
    sk1_snap2 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap4)
    sk1_snap4 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap6)
    sk1_snap6 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap8)
    sk1_snap8 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap10)
    sk1_snap10 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap12)
    sk1_snap12 = spearman_difference(np.amin(anp_array, axis=1))

    sk2_snap2 = spearman_difference(mean_anp_rank - rank_snap2)
    sk2_snap4 = spearman_difference(mean_anp_rank - rank_snap4)
    sk2_snap6 = spearman_difference(mean_anp_rank - rank_snap6)
    sk2_snap8 = spearman_difference(mean_anp_rank - rank_snap8)
    sk2_snap10 = spearman_difference(mean_anp_rank - rank_snap10)
    sk2_snap12 = spearman_difference(mean_anp_rank - rank_snap12)

    counter += 1
    res = (tuple(zavisnost),
           {'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
            'ANP3': anp3.tezine.flatten(), 'ANP4': anp4.tezine.flatten(),
            'min_ANP': min_anp, 'max_ANP': max_anp, 'R1_min': r1_min, 'R1_max': r1_max,
            'SNAP2_elem_R1': snap2_elem_r1, 'SNAP4_elem_R1': snap4_elem_r1,
            'SNAP6_elem_R1': snap6_elem_r1, 'SNAP8_elem_R1': snap8_elem_r1,
            'SNAP10_elem_R1': snap10_elem_r1, 'SNAP12_elem_R1': snap12_elem_r1,
            'R2_min': r2_min, 'R2_max': r2_max,
            'SNAP2_elem_R2': snap2_elem_r2, 'SNAP4_elem_R2': snap4_elem_r2,
            'SNAP6_elem_R2': snap6_elem_r2, 'SNAP8_elem_R2': snap8_elem_r2,
            'SNAP10_elem_R2': snap10_elem_r2, 'SNAP12_elem_R2': snap12_elem_r2,
            'R3_min': r3_min, 'R3_max': r3_max,
            'SNAP2_elem_R3': snap2_elem_r3, 'SNAP4_elem_R3': snap4_elem_r3,
            'SNAP6_elem_R3': snap6_elem_r3, 'SNAP8_elem_R3': snap8_elem_r3,
            'SNAP10_elem_R3': snap10_elem_r3,'SNAP12_elem_R3': snap12_elem_r3,
            # 'rank_ANP1': rank_anp1, 'rank_ANP2': rank_anp2, 'rank_ANP3': rank_anp3,
            # 'rank_ANP4': rank_anp4, 'rank_SNAP1': rank_snap1, 'rank_SNAP2': rank_snap2,
            # 'rank_SNAP3': rank_snap3, 'rank_SNAP4': rank_snap4, 'rank_SNAP5': rank_snap5,
            # 'rank_SNAP6': rank_snap6, 'rank_SNAP7': rank_snap7, 'rank_SNAP8': rank_snap8,
            # 'rank_SNAP9': rank_snap9, 'rank_SNAP10': rank_snap10, 'rank_SNAP11': rank_snap11,
            # 'rank_SNAP12': rank_snap12,
            'sk1_snap2': sk1_snap2, 'sk2_snap2': sk2_snap2, 'sk1_snap4': sk1_snap4, 'sk2_snap4': sk2_snap4,
            'sk1_snap6': sk1_snap6, 'sk2_snap6': sk2_snap6, 'sk1_snap8': sk1_snap8, 'sk2_snap8': sk2_snap8,
            'sk1_snap10': sk1_snap10, 'sk2_snap10': sk2_snap10, 'sk1_snap12': sk1_snap12, 'sk2_snap12': sk2_snap12})
    return res


def calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap1):
    return np.hstack(((rank_anp1 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp2 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp3 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp4 - rank_snap1).reshape((brojKriterija, 1))))


def spearman_difference(diff):
    d2_square = np.square(diff)
    sumD2 = np.sum(d2_square)
    return 1 - ((6 * sumD2) / (math.pow(brojKriterija, 3) - brojKriterija))


def main():
    koraci = {4: 10, 5: 1000000, 6: 100000000000000}
    pool = ThreadPool(int(brojDretvi))
    n = int((brojKriterija - 1) * brojKriterija)
    stop = int(math.pow(5, n) - 1)
    raspon = [0, 1, 2, 3, 4]
    if krnjiDematel:
        raspon = [0, 2, 4]
        stop = int(math.pow(3, n) - 1)
        koraci = {4: 10, 5: 1000, 6: 100000000}
    listaZavisnosti = itertools.islice(
        itertools.product(raspon, repeat=n), 0, stop, koraci[brojKriterija])

    usporedba = list(np.ones(int(n/2)))
    doPartOfSimulation = partial(doSimulation, usporedba)
    results.update(pool.imap_unordered(doPartOfSimulation, listaZavisnosti, chunksize=400))


def processResults():
    df = pd.DataFrame(list(results.values()))
    print("Broj krit. ", brojKriterija, " broj komb:", len(df.index))

    correlation = df[correlationColumns].apply(pd.Series.value_counts)
    correlation.to_csv((outDir + "/spearman_" + str(brojKriterija) + ".csv"), header=True,
                       sep=";",
                       na_rep='0')
    distributions = df[distributionColumns].apply(pd.Series.value_counts)
    distributions.to_csv((outDir + "/distrib_" + str(brojKriterija) + ".csv"), header=True,
                         sep=";",
                         na_rep='0')


if __name__ == "__main__":
    print("Argumenti", sys.argv)
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "-help", "--help"):
        print("main.py <broj_kriterija> <brojDretvi> <direktorijIzlaza> <krnjidematel [0 ili 1]>")
        print("PRIMJER: python main.py ~/usporedbe.csv ~/zavisnosti.csv 4 /shared/mdzeko")
        sys.exit(2)
    brojKriterija, brojDretvi, outDir, krnjiDematel = tuple(sys.argv[1:])
    brojKriterija = int(brojKriterija)
    if brojKriterija not in (4, 5, 6, 7, 8, 9, 10):
        print("Broj kriterija nije postavljen")
        exit(-1)
    results = {}
    start = time.time()
    main()
    processResults()
    print(time.time() - start)
