from anp.ANP import ANP
from anp.MatricaUsporedbi import MatricaUsporedbi
from anp.MatricaZavisnosti import MatricaZavisnosti
from generator.Generetor import Generator
from snap.SNAP import SNAP
import sys
from multiprocessing import Pool as ThreadPool


def ispisTest():
    usp = MatricaUsporedbi(
        Generator.izradiMatUsporedbe([5.0, 5.0, 0.5, 0.5, 0.2, 0.16666666666666666, 1 / 5, 0.5, 0.5, 1], 5))
    zav = MatricaZavisnosti(
        Generator.izradiMatZavisnosti([4, 1, 2, 2, 2, 2, 2, 3, 3, 3, 1, 1, 2, 4, 2, 2, 2, 3, 3, 3], 5))
    snap = SNAP(usp.weights, zav.Z)
    # anp = ANP(usp.weights, zav.Z)
    snap.simulate()
    print(sys.getsizeof(snap))
    snap.printResults()


def generirajUlaz(brKrit):
    gen = Generator(1, brKrit)
    gen.generateAllComparisonMatrices()
    # gen.generateDependancyMatrices()


if __name__ == "__main__":
    generirajUlaz(4)
