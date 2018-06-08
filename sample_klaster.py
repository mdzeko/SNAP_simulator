from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
import numpy as np
from functools import partial
import csv, math
import ast
import time
import pandas as pd
import sys
from multiprocessing import Pool as ThreadPool

totalElapsed = time.process_time()
brojKlastera = 2
brojKriterija = 4


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
            {'zavisnost': zavisnost, 'usporedba': usporedba,
             'ANP1': anp1.tezine.flatten(), 'ANP2': anp2.tezine.flatten(),
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
    df.to_csv((outDir + "/rezANPvsANP_" + str(brojKriterija) + ".csv"), header=True,
              sep=";")
    print("Broj krit. ", brojKriterija, " broj klast. ", brojKlastera, " broj komb:", len(df.index))


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
