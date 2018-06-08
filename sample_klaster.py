from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
import numpy as np
from scipy.stats import rankdata
from functools import partial
import csv, math
import ast
import time
import pandas as pd
import sys, os
from multiprocessing import Pool as ThreadPool

totalElapsed = time.process_time()
brojKlastera = 2
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

distributionColumns = ['SNAP1_elem_R1', 'SNAP2_elem_R1', 'SNAP3_elem_R1', 'SNAP4_elem_R1',
                       'SNAP5_elem_R1', 'SNAP6_elem_R1', 'SNAP7_elem_R1', 'SNAP8_elem_R1',
                       'SNAP9_elem_R1', 'SNAP10_elem_R1', 'SNAP11_elem_R1', 'SNAP12_elem_R1',
                       'SNAP1_elem_R2', 'SNAP2_elem_R2', 'SNAP3_elem_R2', 'SNAP4_elem_R2',
                       'SNAP5_elem_R2', 'SNAP6_elem_R2', 'SNAP7_elem_R2', 'SNAP8_elem_R2',
                       'SNAP9_elem_R2', 'SNAP10_elem_R2', 'SNAP11_elem_R2', 'SNAP12_elem_R2',
                       'SNAP1_elem_R3', 'SNAP2_elem_R3', 'SNAP3_elem_R3', 'SNAP4_elem_R3',
                       'SNAP5_elem_R3', 'SNAP6_elem_R3', 'SNAP7_elem_R3', 'SNAP8_elem_R3',
                       'SNAP9_elem_R3', 'SNAP10_elem_R3', 'SNAP11_elem_R3', 'SNAP12_elem_R3']

correlationColumns = ['sk1_snap1', 'sk2_snap1', 'sk1_snap2', 'sk2_snap2', 'sk1_snap3', 'sk2_snap3', 'sk1_snap4',
                      'sk2_snap4', 'sk1_snap5', 'sk2_snap5', 'sk1_snap6', 'sk2_snap6', 'sk1_snap7', 'sk2_snap7',
                      'sk1_snap8', 'sk2_snap8', 'sk1_snap9', 'sk2_snap9', 'sk1_snap10', 'sk2_snap10', 'sk1_snap11',
                      'sk2_snap11', 'sk1_snap12', 'sk2_snap12']


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

    # Zaokruziti tezine na 5 decimala jer rangiranje zna bit osjetljivo
    rank_anp1 = rankdata(np.round(anp1.tezine, 5), method='ordinal')
    rank_anp2 = rankdata(np.round(anp2.tezine, 5), method='ordinal')
    rank_anp3 = rankdata(np.round(anp3.tezine, 5), method='ordinal')
    rank_anp4 = rankdata(np.round(anp4.tezine, 5), method='ordinal')
    mean_anp_rank = (rank_anp1 + rank_anp2 + rank_anp3 + rank_anp4) / 4
    rank_snap1 = rankdata(np.round(snap.tezine_S1, 5), method='ordinal')
    rank_snap2 = rankdata(np.round(snap.tezine_S2, 5), method='ordinal')
    rank_snap3 = rankdata(np.round(snap.tezine_S3, 5), method='ordinal')
    rank_snap4 = rankdata(np.round(snap.tezine_S4, 5), method='ordinal')
    rank_snap5 = rankdata(np.round(snap.tezine_S5, 5), method='ordinal')
    rank_snap6 = rankdata(np.round(snap.tezine_S6, 5), method='ordinal')
    rank_snap7 = rankdata(np.round(snap.tezine_S7, 5), method='ordinal')
    rank_snap8 = rankdata(np.round(snap.tezine_S8, 5), method='ordinal')
    rank_snap9 = rankdata(np.round(snap.tezine_S9, 5), method='ordinal')
    rank_snap10 = rankdata(np.round(snap.tezine_S10, 5), method='ordinal')
    rank_snap11 = rankdata(np.round(snap.tezine_S11, 5), method='ordinal')
    rank_snap12 = rankdata(np.round(snap.tezine_S12, 5), method='ordinal')

    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap1)
    sk1_snap1 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap2)
    sk1_snap2 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap3)
    sk1_snap3 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap4)
    sk1_snap4 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap5)
    sk1_snap5 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap6)
    sk1_snap6 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap7)
    sk1_snap7 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap8)
    sk1_snap8 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap9)
    sk1_snap9 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap10)
    sk1_snap10 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap11)
    sk1_snap11 = spearman_difference(np.amin(anp_array, axis=1))
    anp_array = calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap12)
    sk1_snap12 = spearman_difference(np.amin(anp_array, axis=1))

    sk2_snap1 = spearman_difference(mean_anp_rank - rank_snap1)
    sk2_snap2 = spearman_difference(mean_anp_rank - rank_snap2)
    sk2_snap3 = spearman_difference(mean_anp_rank - rank_snap3)
    sk2_snap4 = spearman_difference(mean_anp_rank - rank_snap4)
    sk2_snap5 = spearman_difference(mean_anp_rank - rank_snap5)
    sk2_snap6 = spearman_difference(mean_anp_rank - rank_snap6)
    sk2_snap7 = spearman_difference(mean_anp_rank - rank_snap7)
    sk2_snap8 = spearman_difference(mean_anp_rank - rank_snap8)
    sk2_snap9 = spearman_difference(mean_anp_rank - rank_snap9)
    sk2_snap10 = spearman_difference(mean_anp_rank - rank_snap10)
    sk2_snap11 = spearman_difference(mean_anp_rank - rank_snap11)
    sk2_snap12 = spearman_difference(mean_anp_rank - rank_snap12)

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
            # 'rank_ANP1': rank_anp1, 'rank_ANP2': rank_anp2, 'rank_ANP3': rank_anp3,
            # 'rank_ANP4': rank_anp4, 'rank_SNAP1': rank_snap1, 'rank_SNAP2': rank_snap2,
            # 'rank_SNAP3': rank_snap3, 'rank_SNAP4': rank_snap4, 'rank_SNAP5': rank_snap5,
            # 'rank_SNAP6': rank_snap6, 'rank_SNAP7': rank_snap7, 'rank_SNAP8': rank_snap8,
            # 'rank_SNAP9': rank_snap9, 'rank_SNAP10': rank_snap10, 'rank_SNAP11': rank_snap11,
            # 'rank_SNAP12': rank_snap12,
            'sk1_snap1': sk1_snap1, 'sk2_snap1': sk2_snap1, 'sk1_snap2': sk1_snap2,
            'sk2_snap2': sk2_snap2, 'sk1_snap3': sk1_snap3, 'sk2_snap3': sk2_snap3,
            'sk1_snap4': sk1_snap4, 'sk2_snap4': sk2_snap4, 'sk1_snap5': sk1_snap5,
            'sk2_snap5': sk2_snap5, 'sk1_snap6': sk1_snap6, 'sk2_snap6': sk2_snap6,
            'sk1_snap7': sk1_snap7, 'sk2_snap7': sk2_snap7, 'sk1_snap8': sk1_snap8,
            'sk2_snap8': sk2_snap8, 'sk1_snap9': sk1_snap9, 'sk2_snap9': sk2_snap9,
            'sk1_snap10': sk1_snap10, 'sk2_snap10': sk2_snap10, 'sk1_snap11': sk1_snap11,
            'sk2_snap11': sk2_snap11, 'sk1_snap12': sk1_snap12, 'sk2_snap12': sk2_snap12})
    return res


def doClusterTest(usporedba, zavisnost):
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

    k_anp1 = ANP(usp.weights, zav.Z)
    k_anp1.simulate(variant='klasters')

    k_anp2 = ANP(usp.weights, zav.Z)
    k_anp2.simulate(fiktivnaAlt=True, matricaPrijelaza=False, variant='klasters')

    k_anp3 = ANP(usp.weights, zav.Z)
    k_anp3.simulate(fiktivnaAlt=False, matricaPrijelaza=True, variant='klasters')

    k_anp4 = ANP(usp.weights, zav.Z)
    k_anp4.simulate(matricaPrijelaza=True, fiktivnaAlt=True, variant='klasters')

    dif1 = np.round(np.abs(anp1.tezine.flatten() - k_anp1.tezine.flatten()), 5)
    dif2 = np.round(np.abs(anp2.tezine.flatten() - k_anp2.tezine.flatten()), 5)
    dif3 = np.round(np.abs(anp3.tezine.flatten() - k_anp3.tezine.flatten()), 5)
    dif4 = np.round(np.abs(anp4.tezine.flatten() - k_anp4.tezine.flatten()), 5)

    avg1 = np.average(dif1)
    avg2 = np.average(dif2)
    avg3 = np.average(dif3)
    avg4 = np.average(dif4)

    return ((zavisnost, usporedba),
            {'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
             'ANP3': anp3.tezine.flatten(), 'ANP4': anp4.tezine.flatten(),
             'K_ANP1': k_anp1.tezine.flatten(), 'K_ANP2': k_anp2.tezine.flatten(),
             'K_ANP3': k_anp3.tezine.flatten(), 'K_ANP4': k_anp4.tezine.flatten(),
             'diff1': dif1, 'diff2': dif2, 'diff3': dif3, 'diff4': dif4,
             'avg1': avg1, 'avg2': avg2, 'avg3': avg3, 'avg4': avg4})


def calculate_diff_array(rank_anp1, rank_anp2, rank_anp3, rank_anp4, rank_snap1):
    return np.hstack(((rank_anp1 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp2 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp3 - rank_snap1).reshape((brojKriterija, 1)),
                      (rank_anp4 - rank_snap1).reshape((brojKriterija, 1))))


def spearman_difference(diff):
    d2_square = np.square(diff)
    sumD2 = np.sum(d2_square)
    return 1 - ((6 * sumD2) / (math.pow(brojKriterija, 3) - brojKriterija))


def clear():
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        os.system('clear')
    else:
        os.system('cls')


def main():
    pool = ThreadPool(int(brojDretvi))
    # execute only if run as a script
    with open(inputUsporedbe, "r") as csvUsporedbe, open(inputZavisnosti, "r") as csvZavisnosti:
        usporedbeReader = csv.reader(csvUsporedbe, delimiter=';')
        zavisnostiReader = csv.reader(csvZavisnosti, delimiter=';')
        listaZavisnosti = list(zavisnostiReader)
        listaUsporedbi = list(usporedbeReader)

    for usporedba in listaUsporedbi:
        doPartOfSimulation = partial(doClusterTest, usporedba[0])
        results.update(pool.imap_unordered(doPartOfSimulation, [redak[0] for redak in listaZavisnosti], chunksize=400))


def processResults():
    df = pd.DataFrame(list(results.values()))
    print("Broj krit. ", brojKriterija, " broj klast. ", brojKlastera, " broj komb:", len(df.index))

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
        print("main.py <datotekaUsporedbi> <datotekaZavisnost> <brojDretvi> <direktorijIzlaza>")
        print("PRIMJER: python main.py '~/usporedbe.csv' '~/zavisnosti.csv' 4 '/shared/mdzeko'")
        sys.exit(2)
    inputUsporedbe, inputZavisnosti, brojDretvi, outDir = tuple(sys.argv[1:])
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
